"""
RabbitMQ 任务调度器模块

提供 SimNIBS RabbitMQ 服务的任务调度功能，包括：
- 任务消息处理（model / forward / inverse / ack_test）
- 工作线程管理（每个任务在独立线程中执行）
- 进度汇报（通过消息队列发送进度到后端）

架构说明
---------
1. run_service() 启动两个组件：
   - RabbitMQPublisher（发送线程）：从 message_queue 取消息发送到 simnibs_to_backend 队列
   - RabbitMQConsumer（监听线程）：监听 backend_to_simnibs 队列，收到消息后触发回调

2. 消息处理流程：
   handle_message() 收到消息 -> 验证参数 -> 启动工作线程 -> execute_task() 执行任务
                                                      |
                                                      v
                                          任务完成后调用 schedule_ack() 发送 ack

3. 任务类型：
   - model：CHARM 头模生成（支持断点续传）
   - forward：TI 正向仿真
   - inverse：TI 逆向仿真（优化）
   - ack_test：ACK 时机测试
"""

import json
import logging
import os
import re
import shutil
import threading
import time
from functools import partial
from pathlib import Path
from queue import Queue
from typing import Any

from neuracle.atlas.standardized import get_standardized_roi_path
from neuracle.charm import (
    create_mesh,
    create_surfaces,
    denoise_inputs,
    init_atlas,
    prepare_t1,
    prepare_t2,
    run_segmentation,
)
from neuracle.rabbitmq.consumer import RabbitMQConsumer
from neuracle.rabbitmq.message_builder import build_progress_message
from neuracle.rabbitmq.params import (
    dict_to_ack_test_params,
    dict_to_forward_params,
    dict_to_inverse_params,
    dict_to_model_params,
)
from neuracle.rabbitmq.progress import (
    ForwardProgress,
    InverseProgress,
    ModelProgress,
    load_progress,
    save_progress,
)
from neuracle.rabbitmq.publisher import RabbitMQPublisher
from neuracle.rabbitmq.schemas import (
    AckTestParams,
    ForwardParams,
    InverseParams,
    ModelParams,
)
from neuracle.rabbitmq.task_handlers import schedule_ack
from neuracle.rabbitmq.validator import (
    ValidationError,
    validate_ack_test_params,
    validate_forward_params,
    validate_inverse_params,
    validate_model_params,
)
from neuracle.storage.oss import (
    download_folder_from_oss,
    download_input_file,
    upload_model_outputs,
    upload_task_result,
)
from neuracle.storage.paths import (
    ensure_data_root,
    get_model_mesh_path,
    get_subject_dir,
    get_task_output_dir,
    normalize_dir_path,
    reset_task_output_dir,
)
from neuracle.ti_optimization import (
    get_electrode_mapping,
    init_optimization,
    run_optimization,
    setup_electrodes_and_roi,
    setup_goal,
)
from neuracle.ti_simulation import (
    calculate_ti,
    run_tdcs_simulation,
    setup_electrode_pair1,
    setup_electrode_pair2,
    setup_session,
)
from neuracle.utils import (
    DEBUG,
    N_WORKERS,
    NON_ROI_THRESHOLD,
    cond_dict_to_list,
    find_montage_file,
)
from neuracle.utils.ti_export import export_ti_to_nifti

logger = logging.getLogger(__name__)


def send_progress(
    message_queue: Queue,
    task_id: str,
    msg_type: str,
    progress_rate: int,
    error_message: str | None = None,
    result: dict | None = None,
) -> None:
    """
    发送进度消息到 simnibs_to_backend 队列

    Parameters
    ----------
    message_queue : Queue
        消息队列
    task_id : str
        任务 ID
    msg_type : str
        任务类型
    progress_rate : int
        进度百分比
    error_message : str, optional
        错误消息
    result : dict, optional
        完成结果
    """
    progress_msg = build_progress_message(
        id=task_id,
        msg_type=msg_type,
        progress_rate=progress_rate,
        message=error_message,
        result=result,
    )
    message_queue.put(progress_msg)
    logger.debug("发送进度消息: %s %d%%", task_id, progress_rate)


def handle_model_task(
    message_queue: Queue, task_id: str, params: ModelParams, redelivered: bool = False
) -> None:
    """
    处理 CHARM 头模生成任务

    任务流程（步步为营，每步完成都会保存进度并通知后端）
    ------
    步骤 0：初始化
        - 创建 subject 目录
        - 判断是否为重发消息（redelivered），若是则从 .progress.txt 恢复进度
        - 新消息则从头开始，发送进度 0% 表示任务启动

    步骤 1：下载输入文件（0%~10%）
        - 从 OSS 下载 T1/T2/DTI 图像文件到本地 subject 目录
        - 仅在 PREPARE_T1_DONE 未完成时执行（支持断点续传）

    步骤 2：T1 图像预处理（10%）
        - 调用 prepare_t1() 进行 T1 图像格式处理和配准
        - 若失败，尝试强制使用 qform

    步骤 3：T2 图像预处理（20%）
        - 调用 prepare_t2() 进行 T2 图像处理
        - 若失败，尝试强制使用 sform

    步骤 4：去噪处理（35%）
        - 调用 denoise_inputs() 对图像进行去噪

    步骤 5：Atlas 初始化（50%）
        - 调用 init_atlas() 初始化标准 Atlas

    步骤 6：组织分割（70%）
        - 调用 run_segmentation() 进行灰质/白质/CSF 等组织分割

    步骤 7：表面重建（85%）
        - 调用 create_surfaces() 生成头皮、脑皮层等表面 mesh

    步骤 8：Mesh 生成与上传（100%）
        - 调用 create_mesh() 生成最终的头部模型 mesh 文件
        - 上传到 OSS
        - 删除本地进度文件，发送完成消息

    Parameters
    ----------
    message_queue : Queue
        消息队列，用于发送进度更新
    task_id : str
        任务 ID
    params : ModelParams
        头模生成参数
    redelivered : bool
        消息是否为 RabbitMQ 重发（支持进度恢复：当 redelivered=True 时从进度文件恢复）
    """
    logger.info(
        "收到 Model 任务: task_id=%s, dir_path=%s, T1_file=%s, T2_file=%s, DTI_file=%s",
        task_id,
        params.dir_path,
        params.T1_file_path,
        params.T2_file_path,
        params.DTI_file_path,
    )

    # ===== 步骤 0：初始化 =====
    # 创建 subject 本地目录，若不存在则新建
    subject_dir = get_subject_dir(params.dir_path)
    ensure_data_root()
    subject_dir.mkdir(parents=True, exist_ok=True)

    # 进度文件用于断点续传（若 RabbitMQ 重发消息，可从上次进度继续）
    progress_file = subject_dir / ".progress.txt"
    if redelivered:
        # 消息为重发，从本地进度文件恢复执行进度
        current_progress = load_progress(progress_file)
        logger.info("任务 %s 为重发消息，从进度 %d 恢复", task_id, current_progress)
    else:
        # 新任务，删除旧进度文件，从 0 开始
        if progress_file.exists():
            progress_file.unlink()
        current_progress = 0
        logger.info("任务 %s 为新消息，从头开始执行", task_id)
        # 发送任务启动进度 0%
        send_progress(message_queue, task_id, "model", ModelProgress.START)

    # ===== 步骤 1：下载输入文件 =====
    # 准备 T1/T2/DTI 文件本地路径
    t1_local_path = subject_dir / Path(params.T1_file_path).name
    t2_local_path = (
        subject_dir / Path(params.T2_file_path).name if params.T2_file_path else None
    )

    # 仅在尚未完成 T1 准备时下载（避免重复下载）
    if current_progress < ModelProgress.PREPARE_T1_DONE:
        # 从 OSS 下载输入文件到本地
        t1_local_path = download_input_file(params.T1_file_path, t1_local_path)
        if params.T2_file_path:
            t2_local_path = download_input_file(params.T2_file_path, t2_local_path)
        if params.DTI_file_path:
            download_input_file(
                params.DTI_file_path,
                subject_dir / Path(params.DTI_file_path).name,
            )

    # ===== 步骤 2：T1 图像预处理 =====
    if current_progress < ModelProgress.PREPARE_T1_DONE:
        if t1_local_path and t1_local_path.exists():
            try:
                # T1 图像格式转换和配准（默认方式）
                prepare_t1(str(subject_dir), str(t1_local_path))
            except Exception:
                try:
                    # 失败时尝试强制使用 qform
                    prepare_t1(str(subject_dir), str(t1_local_path), force_qform=True)
                except Exception:
                    # 再尝试强制使用 sform
                    prepare_t1(str(subject_dir), str(t1_local_path), force_sform=True)
        # 保存进度并通知后端
        save_progress(progress_file, ModelProgress.PREPARE_T1_DONE)
        send_progress(message_queue, task_id, "model", ModelProgress.PREPARE_T1_DONE)

    # ===== 步骤 3：T2 图像预处理 =====
    if current_progress < ModelProgress.PREPARE_T2_DONE:
        if t2_local_path and t2_local_path.exists():
            try:
                # T2 图像格式转换并配准到 T1
                prepare_t2(str(subject_dir), str(t2_local_path), register_t2=True)
            except Exception:
                try:
                    # 失败时尝试强制使用 qform
                    prepare_t2(
                        str(subject_dir),
                        str(t2_local_path),
                        register_t2=True,
                        force_qform=True,
                    )
                except Exception:
                    # 再尝试强制使用 sform
                    prepare_t2(
                        str(subject_dir),
                        str(t2_local_path),
                        register_t2=True,
                        force_sform=True,
                    )
        save_progress(progress_file, ModelProgress.PREPARE_T2_DONE)
        send_progress(message_queue, task_id, "model", ModelProgress.PREPARE_T2_DONE)

    # ===== 步骤 4：去噪处理 =====
    if current_progress < ModelProgress.DENOISE_DONE:
        denoise_inputs(str(subject_dir))
        save_progress(progress_file, ModelProgress.DENOISE_DONE)
        send_progress(message_queue, task_id, "model", ModelProgress.DENOISE_DONE)

    # ===== 步骤 5：Atlas 初始化 =====
    if current_progress < ModelProgress.INIT_ATLAS_DONE:
        init_atlas(str(subject_dir))
        save_progress(progress_file, ModelProgress.INIT_ATLAS_DONE)
        send_progress(message_queue, task_id, "model", ModelProgress.INIT_ATLAS_DONE)

    # ===== 步骤 6：组织分割 =====
    if current_progress < ModelProgress.SEGMENTATION_DONE:
        run_segmentation(str(subject_dir))
        save_progress(progress_file, ModelProgress.SEGMENTATION_DONE)
        send_progress(message_queue, task_id, "model", ModelProgress.SEGMENTATION_DONE)

    # ===== 步骤 7：表面重建 =====
    if current_progress < ModelProgress.SURFACES_DONE:
        create_surfaces(str(subject_dir))
        save_progress(progress_file, ModelProgress.SURFACES_DONE)
        send_progress(message_queue, task_id, "model", ModelProgress.SURFACES_DONE)

    # ===== 步骤 8：Mesh 生成与上传 =====
    if current_progress < ModelProgress.COMPLETED:
        # 生成最终头部模型 mesh 文件
        create_mesh(str(subject_dir))
        # 提取 subid 用于构造动态文件名
        subid = re.search("m2m_(.+)", params.dir_path).group(1)
        # 规范化路径后上传到 OSS
        normalized = normalize_dir_path(params.dir_path)
        upload_model_outputs(params.dir_path, subject_dir, normalized, subid)
        # 标记完成，删除进度文件
        save_progress(progress_file, ModelProgress.COMPLETED)
        progress_file.unlink()
        # 构造返回结果，包含 mesh 文件的 OSS 路径
        msh_path = f"{normalize_dir_path(params.dir_path)}/{subid}.msh"
        result = {"msh_file_path": msh_path}
        send_progress(
            message_queue, task_id, "model", ModelProgress.COMPLETED, result=result
        )


def handle_forward_task(
    message_queue: Queue, task_id: str, params: ForwardParams, redelivered: bool = False
) -> None:
    """
    处理 TI 正向仿真任务

    正向仿真用于计算给定电极配置下的电场分布。
    与逆向仿真不同，正向仿真是"已知电极电流，求解电场"，不涉及优化。

    任务流程
    ------
    步骤 0：初始化与下载（0%）
        - 获取 subject 目录，若本地无则从 OSS 下载
        - 创建任务输出目录

    步骤 1：会话初始化（10%）
        - 调用 setup_session() 初始化 SimNIBS 仿真会话
        - 配置各向异性类型、电导率、张量文件、电极帽等

    步骤 2：电极对 A 配置（20%）
        - 调用 setup_electrode_pair1() 配置第一组电极的电流

    步骤 3：电极对 B 配置（35%）
        - 调用 setup_electrode_pair2() 配置第二组电极的电流

    步骤 4：TDCS 仿真执行（70%）
        - 调用 run_tdcs_simulation() 并行执行两组电极的仿真
        - 返回两个 mesh 文件（分别对应电极对 A 和 B）

    步骤 5：TI 计算（85%）
        - 调用 calculate_ti() 计算两个电场的矢量叠加（TI）
        - TI = E1 + E2（考虑方向性）

    步骤 6：NIfTI 导出（95%）
        - 调用 export_ti_to_nifti() 将 TI 结果导出为 NIfTI 格式
        - 以 T1 图像为参考空间

    步骤 7：上传与清理（100%）
        - 上传结果到 OSS
        - 删除本地临时输出目录

    Parameters
    ----------
    message_queue : Queue
        消息队列
    task_id : str
        任务 ID
    params : ForwardParams
        正向仿真参数
    redelivered : bool
        消息是否为 RabbitMQ 重发（当前未使用，Forward 任务暂不支持进度恢复）
    """
    logger.info(
        "收到 Forward 任务: task_id=%s, dir_path=%s, montage=%s, anisotropy=%s, "
        "electrode_A=%s, electrode_B=%s, conductivity=%s",
        task_id,
        params.dir_path,
        params.montage,
        params.anisotropy,
        params.electrode_A,
        params.electrode_B,
        params.conductivity_config,
    )

    # ===== 步骤 0：初始化与下载 =====
    send_progress(message_queue, task_id, "forward", ForwardProgress.START)

    # 获取 subject 本地目录，若不存在则从 OSS 下载完整目录
    subject_dir = get_subject_dir(params.dir_path)
    if not subject_dir.is_dir():
        ensure_data_root()
        subject_dir.mkdir(parents=True, exist_ok=True)
        download_folder_from_oss(normalize_dir_path(params.dir_path), subject_dir)

    # 创建任务输出目录（TI_simulation/{task_id}）
    output_dir = get_task_output_dir(params.dir_path, "TI_simulation", task_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定各向异性类型：vn=体积中性（各向异性），scalar=标量（各向同性）
    anisotropy_type = "vn" if params.anisotropy else "scalar"

    # 非 DEBUG 模式下重置输出目录（清空旧文件）
    if not DEBUG:
        reset_task_output_dir(str(output_dir))

    # 查找 EEG 电极导联文件
    eeg_cap = find_montage_file(str(subject_dir), params.montage)
    # 获取头部模型 mesh 文件路径
    mesh_path = get_model_mesh_path(params.dir_path)

    # 提取电极名称和电流值（mA 转为 A）
    electrode_A_names = [e.name for e in params.electrode_A]
    electrode_A_currents = [e.current_mA / 1000 for e in params.electrode_A]
    electrode_B_names = [e.name for e in params.electrode_B]
    electrode_B_currents = [e.current_mA / 1000 for e in params.electrode_B]

    # ===== 步骤 1：会话初始化 =====
    S = setup_session(
        subject_dir=str(subject_dir),
        msh_file_path=str(mesh_path),
        output_dir=str(output_dir),
        anisotropy_type=anisotropy_type,
        cond=cond_dict_to_list(params.conductivity_config),
        fname_tensor=params.DTI_file_path,
        eeg_cap=eeg_cap,
    )
    send_progress(message_queue, task_id, "forward", ForwardProgress.SESSION_SETUP)

    # ===== 步骤 2：电极对 A 配置 =====
    setup_electrode_pair1(
        session=S,
        electrode_pair1=electrode_A_names,
        current1=electrode_A_currents,
    )
    send_progress(
        message_queue, task_id, "forward", ForwardProgress.ELECTRODE_PAIR1_DONE
    )

    # ===== 步骤 3：电极对 B 配置 =====
    setup_electrode_pair2(
        session=S,
        electrode_pair2=electrode_B_names,
        current2=electrode_B_currents,
    )
    send_progress(
        message_queue, task_id, "forward", ForwardProgress.ELECTRODE_PAIR2_DONE
    )

    # ===== 步骤 4：TDCS 仿真执行 =====
    # 并行求解两组电极下的电场分布，返回各自的 mesh 文件路径
    mesh1_path, mesh2_path = run_tdcs_simulation(
        session=S,
        subject_dir=str(subject_dir),
        output_dir=str(output_dir),
        n_workers=N_WORKERS,
    )
    send_progress(
        message_queue, task_id, "forward", ForwardProgress.TDCS_SIMULATION_DONE
    )

    # ===== 步骤 5：TI 计算 =====
    # 叠加两个电场计算瞬态干涉（Temporal Interference）场
    ti_mesh_path = calculate_ti(
        mesh1_path=mesh1_path,
        mesh2_path=mesh2_path,
        output_dir=str(output_dir),
    )
    send_progress(message_queue, task_id, "forward", ForwardProgress.TI_CALCULATED)

    # ===== 步骤 6：NIfTI 导出 =====
    # 将 TI mesh 配准到 T1 空间并导出为 NIfTI
    subid = re.search("m2m_(.+)", params.dir_path).group(1)
    ti_nifti_path = export_ti_to_nifti(
        msh_path=ti_mesh_path,
        output_dir=str(output_dir),
        reference=str(subject_dir / Path(params.T1_file_path).name),
        field_name="max_TI",
        prefix=f"{subid}_simulation",
    )
    send_progress(message_queue, task_id, "forward", ForwardProgress.NIFTI_EXPORTED)

    # ===== 步骤 7：上传与清理 =====
    # 上传 NIfTI 到 OSS，路径格式：{dir_path}_TI_simulation_{task_id}/{subid}_simulation_max_TI.nii.gz
    # 注意：export_ti_to_nifti 生成的文件名格式为 {prefix}_{field_name}.nii.gz
    oss_folder = f"{normalize_dir_path(params.dir_path)}_TI_simulation_{task_id}"
    ti_file_key = upload_task_result(
        Path(ti_nifti_path),
        f"{oss_folder}/{subid}_simulation_max_TI.nii.gz",
    )
    # 删除本地临时输出目录（非 DEBUG 模式）
    if not DEBUG:
        shutil.rmtree(output_dir, ignore_errors=True)

    result = {"TI_file": ti_file_key}
    send_progress(
        message_queue, task_id, "forward", ForwardProgress.COMPLETED, result=result
    )


def handle_inverse_task(
    message_queue: Queue, task_id: str, params: InverseParams, redelivered: bool = False
) -> None:
    """
    处理 TI 逆向仿真任务（优化任务）

    逆向仿真是"已知目标电场分布，反求电极电流配置"的优化过程。
    与正向仿真不同，逆向仿真需要迭代优化以找到最优电流配置。

    任务流程
    ------
    步骤 0：初始化与下载（0%）
        - 获取 subject 目录，若本地无则从 OSS 下载
        - 创建任务输出目录
        - 解析 ROI 参数（atlas 或 MNI 球形区域）

    步骤 1：优化器初始化（10%）
        - 调用 init_optimization() 创建优化器实例
        - 配置头部模型、各向异性、电导率等

    步骤 2：目标设置（20%）
        - 调用 setup_goal() 设置优化目标为 focality（聚焦性）
        - 配置 ROI 内目标电场阈值和非 ROI 区域阈值

    步骤 3：电极与 ROI 配置（35%）
        - 调用 setup_electrodes_and_roi() 配置初始电流和 ROI 区域
        - ROI 可由 atlas 掩码或 MNI 球形区域定义

    步骤 4：执行优化（85%）
        - 调用 run_optimization() 迭代搜索最优电流配置
        - 优化目标：使 ROI 内电场尽量高，非 ROI 内电场尽量低

    步骤 5：电极映射获取（90%）
        - 调用 get_electrode_mapping() 获取优化后的电极-电流映射

    步骤 6：NIfTI 导出（95%）
        - 将优化后的电场分布导出为 NIfTI 格式

    步骤 7：上传与清理（100%）
        - 上传结果到 OSS，包含 TI 文件和电极配置
        - 删除本地临时输出目录

    Parameters
    ----------
    message_queue : Queue
        消息队列
    task_id : str
        任务 ID
    params : InverseParams
        逆向仿真参数
    redelivered : bool
        消息是否为 RabbitMQ 重发（当前未使用，Inverse 任务暂不支持进度恢复）
    """
    # 构造 ROI 信息日志（用于调试和日志记录）
    roi_info = (
        f"atlas={params.roi_param.atlas_param}"
        if params.roi_param.atlas_param
        else f"mni={params.roi_param.mni_param}"
        if params.roi_param.mni_param
        else "None"
    )
    logger.info(
        "收到 Inverse 任务: task_id=%s, dir_path=%s, montage=%s, anisotropy=%s, "
        "current_A=%s, current_B=%s, roi_type=%s, roi=%s, target_threshold=%s, conductivity=%s",
        task_id,
        params.dir_path,
        params.montage,
        params.anisotropy,
        params.current_A,
        params.current_B,
        params.roi_type,
        roi_info,
        params.target_threshold,
        params.conductivity_config,
    )

    # ===== 步骤 0：初始化与下载 =====
    send_progress(message_queue, task_id, "inverse", InverseProgress.START)

    # 获取 subject 本地目录，若不存在则从 OSS 下载完整目录
    subject_dir = get_subject_dir(params.dir_path)
    if not subject_dir.is_dir():
        ensure_data_root()
        subject_dir.mkdir(parents=True, exist_ok=True)
        download_folder_from_oss(normalize_dir_path(params.dir_path), subject_dir)

    # 创建任务输出目录（TI_optimization/{task_id}）
    output_dir = get_task_output_dir(params.dir_path, "TI_optimization", task_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定各向异性类型：vn=体积中性（各向异性），scalar=标量（各向同性）
    anisotropy_type = "vn" if params.anisotropy else "scalar"

    # 非 DEBUG 模式下重置输出目录
    if not DEBUG:
        reset_task_output_dir(str(output_dir))

    # 查找 EEG 电极导联文件
    net_electrode_file = find_montage_file(str(subject_dir), params.montage)
    # 获取头部模型 mesh 文件路径
    mesh_path = get_model_mesh_path(params.dir_path)

    # ROI 参数初始化
    roi_center = None
    roi_radius = None
    roi_center_space = "subject"  # 默认在个体原始空间
    roi_mask_path = None
    roi_mask_space = None
    focality_threshold = params.target_threshold

    # 根据 roi_type 解析 ROI 参数
    if params.roi_type == "atlas" and params.roi_param.atlas_param:
        # 使用标准化 Atlas ROI 掩码（MNI 空间）
        roi_mask_path = str(
            get_standardized_roi_path(
                params.roi_param.atlas_param.name,
                params.roi_param.atlas_param.area,
            )
        )
        if not os.path.exists(roi_mask_path):
            raise FileNotFoundError(
                f"标准化 ROI 不存在: {roi_mask_path}。请先运行 atlas 标准化和 ROI 生成脚本。"
            )
        roi_mask_space = "mni"
    elif params.roi_type == "mni_pos" and params.roi_param.mni_param:
        # 使用 MNI 球形区域作为 ROI
        roi_center = params.roi_param.mni_param.center
        roi_radius = params.roi_param.mni_param.radius
        roi_center_space = "mni"

    # ===== 步骤 1：优化器初始化 =====
    opt = init_optimization(
        subject_dir=str(subject_dir),
        msh_file_path=str(mesh_path),
        output_dir=str(output_dir),
        anisotropy_type=anisotropy_type,
        cond=cond_dict_to_list(params.conductivity_config),
        fname_tensor=params.DTI_file_path,
    )
    send_progress(message_queue, task_id, "inverse", InverseProgress.OPTIMIZATION_INIT)

    # ===== 步骤 2：目标设置 =====
    # 优化目标：聚焦性（focality）
    # - ROI 内电场 >= focality_threshold
    # - 非 ROI 区域电场 <= NON_ROI_THRESHOLD
    setup_goal(
        opt=opt,
        goal="focality",
        focality_threshold=[focality_threshold, NON_ROI_THRESHOLD],
        net_electrode_file=net_electrode_file,
    )
    send_progress(message_queue, task_id, "inverse", InverseProgress.GOAL_SETUP)

    # ===== 步骤 3：电极与 ROI 配置 =====
    setup_electrodes_and_roi(
        opt=opt,
        goal="focality",
        mesh_file_path=mesh_path,
        electrode_current1=[c / 1000 for c in params.current_A],  # mA 转为 A
        electrode_current2=[c / 1000 for c in params.current_B],
        roi_center=roi_center,
        roi_radius=roi_radius,
        roi_center_space=roi_center_space,
        roi_mask_path=roi_mask_path,
        roi_mask_space=roi_mask_space,
    )
    send_progress(
        message_queue, task_id, "inverse", InverseProgress.ELECTRODES_ROI_SETUP
    )

    # ===== 步骤 4：执行优化 =====
    # 启动多进程迭代优化，搜索最优电极电流配置
    run_optimization(opt=opt, n_workers=N_WORKERS)
    send_progress(message_queue, task_id, "inverse", InverseProgress.OPTIMIZATION_DONE)

    # ===== 步骤 5：电极映射获取 =====
    # 读取优化结果，获取最终电极-电流映射关系
    electrode_A, electrode_B = get_electrode_mapping(output_dir=str(output_dir))
    send_progress(
        message_queue, task_id, "inverse", InverseProgress.ELECTRODE_MAPPING_DONE
    )

    # ===== 步骤 6：NIfTI 导出 =====
    # 使用优化后的 mesh 导出电场分布到 NIfTI
    subid = re.search("m2m_(.+)", params.dir_path).group(1)
    msh_name = f"{subid}_tes_mapped_opt_head_mesh.msh"
    msh_path = str(output_dir / "mapped_electrodes_simulation" / msh_name)
    ti_nifti_path = export_ti_to_nifti(
        msh_path=msh_path,
        output_dir=str(output_dir),
        reference=str(subject_dir / Path(params.T1_file_path).name),
        field_name="max_TI",
        prefix=f"{subid}_optimization",
    )
    send_progress(message_queue, task_id, "inverse", InverseProgress.NIFTI_EXPORTED)

    # ===== 步骤 7：上传与清理 =====
    # 上传 NIfTI 和电极配置到 OSS，路径格式：{dir_path}_TI_optimization_{task_id}/{subid}_optimization_max_TI.nii.gz
    # 注意：export_ti_to_nifti 生成的文件名格式为 {prefix}_{field_name}.nii.gz
    oss_folder = f"{normalize_dir_path(params.dir_path)}_TI_optimization_{task_id}"
    ti_file_key = upload_task_result(
        Path(ti_nifti_path),
        f"{oss_folder}/{subid}_optimization_max_TI.nii.gz",
    )
    # 删除本地临时输出目录（非 DEBUG 模式）
    if not DEBUG:
        shutil.rmtree(output_dir, ignore_errors=True)

    # 返回结果包含 TI 文件路径和优化后的电极配置
    result = {
        "TI_file": ti_file_key,
        "electrode_A": electrode_A,
        "electrode_B": electrode_B,
    }
    send_progress(
        message_queue, task_id, "inverse", InverseProgress.COMPLETED, result=result
    )


def handle_ack_test_task(
    message_queue: Queue, task_id: str, params: AckTestParams, redelivered: bool = False
) -> None:
    """
    处理 ack 时机验证任务（用于测试 RabbitMQ ack 机制）

    该任务用于验证消息确认机制的正确性。
    模拟一个耗时任务，在开始和结束时发送进度更新。

    任务流程
    ------
    步骤 0：任务启动（0%）
        - 发送进度 0% 表示任务已开始

    步骤 1：模拟耗时操作
        - sleep 指定的秒数（模拟长任务执行）

    步骤 2：任务完成（100%）
        - 发送进度 100% 和结果

    Parameters
    ----------
    message_queue : Queue
        消息队列
    task_id : str
        任务 ID
    params : AckTestParams
        ack 测试参数
    """
    # ===== 步骤 0：任务启动 =====
    send_progress(message_queue, task_id, "ack_test", 0)
    logger.info(
        "worker_started: task_id=%s type=ack_test sleep_seconds=%.2f",
        task_id,
        params.sleep_seconds,
    )

    # ===== 步骤 1：模拟耗时操作 =====
    time.sleep(params.sleep_seconds)
    logger.info("worker_finished: task_id=%s type=ack_test", task_id)

    # ===== 步骤 2：任务完成 =====
    result = {"sleep_seconds": params.sleep_seconds}
    send_progress(message_queue, task_id, "ack_test", 100, result=result)


def execute_task(
    message_queue: Queue,
    task_id: str,
    msg_type: str,
    task_handler: Any,
    params: Any,
    channel: Any,
    delivery_tag: int,
    redelivered: bool = False,
) -> None:
    """
    在工作线程中执行耗时任务，避免阻塞 RabbitMQ 回调线程

    设计原理
    ------
    RabbitMQ 的 basic_ack() 必须在消费线程（而非工作线程）中调用。
    因此采用以下流程：
    1. handle_message() 在收到消息后，立即启动工作线程执行任务
    2. 工作线程调用 task_handler() 执行实际任务
    3. 任务完成后，通过 schedule_ack() 将 ack 请求投递回消费线程执行

    异常处理
    ------
    - ValidationError：参数验证失败，记录日志，发送错误进度，标记 ack_reason
    - Exception：其他异常，记录日志，发送错误进度，标记 ack_reason
    - finally：无论成功还是失败，都必须调用 schedule_ack() 确保消息被确认

    Parameters
    ----------
    message_queue : Queue
        消息队列
    task_id : str
        任务 ID
    msg_type : str
        任务类型
    task_handler : callable
        任务处理函数
    params : Any
        任务参数
    channel : Any
        RabbitMQ 通道
    delivery_tag : int
        消息 delivery tag（用于 ack）
    redelivered : bool
        消息是否为重发
    """
    ack_reason = "task_finished"  # 默认成功
    try:
        # 调用对应的任务处理函数（handle_model_task / handle_forward_task / ...）
        task_handler(message_queue, task_id, params, redelivered)
        logger.info("任务完成: %s", task_id)
    except ValidationError as e:
        # 参数验证失败（如缺少必填字段、字段类型错误、值范围超限）
        ack_reason = "task_validation_error"
        logger.error("参数验证失败: %s - %s", task_id, e)
        send_progress(message_queue, task_id, msg_type, 0, str(e))
    except Exception as e:
        # 其他执行异常（如文件不存在、OSS 上传失败、SimNIBS 计算错误）
        ack_reason = "task_exception"
        logger.error("任务执行失败: %s - %s", task_id, e)
        send_progress(message_queue, task_id, msg_type, 0, str(e))
    finally:
        # 必须调用 schedule_ack()，确保消息最终被 ack
        # 不能在工作线程直接调 basic_ack()，必须通过 connection.add_callback_threadsafe()
        schedule_ack(channel, delivery_tag, task_id, msg_type, ack_reason)


def handle_message(
    channel: Any, method: Any, properties: Any, body: bytes, message_queue: Queue
) -> None:
    """
    处理接收到的任务消息（RabbitMQ 回调函数）

    消息格式
    ------
    {
        "id": "任务唯一标识符",
        "type": "model | forward | inverse | ack_test",
        "params": { ... }  // 任务参数
    }

    处理流程
    ------
    步骤 1：解析消息
        - 将 JSON bytes 反序列化为 dict
        - 提取 task_id、msg_type、params_data

    步骤 2：参数验证
        - 根据 msg_type 调用对应的 validate_xxx_params() 函数
        - 验证必填字段、字段类型、值范围等

    步骤 3：参数转换
        - 将 params_data 字典转换为对应的 dataclass（ModelParams / ForwardParams / ...）

    步骤 4：启动工作线程
        - 创建 daemon 线程执行 execute_task()
        - execute_task() 会调用具体的 task_handler

    注意
    ----
    - 此函数在 RabbitMQ 消费线程中被调用，必须快速返回
    - 耗时的任务处理在工作线程中进行，不阻塞消费线程
    - 验证失败或解析失败时，直接 ack 丢弃该消息（不重发）

    Parameters
    ----------
    channel : pika.channel.Channel
        RabbitMQ 通道
    method : pika.spec.Basic.Deliver
        消息传递信息（包含 delivery_tag 和 redelivered）
    properties : pika.spec.BasicProperties
        消息属性
    body : bytes
        消息内容（JSON 格式的 bytes）
    message_queue : Queue
        消息队列
    """
    task_id = ""
    msg_type = ""
    # delivery_tag 用于后续 ack，redelivered 表示消息是否被 RabbitMQ 重发
    delivery_tag = method.delivery_tag
    redelivered = method.redelivered

    try:
        # ===== 步骤 1：解析消息 =====
        data = json.loads(body)
        task_id = data["id"]
        msg_type = data["type"]
        params_data = data["params"]

        logger.info(
            "收到任务: %s, 类型: %s, redelivered: %s", task_id, msg_type, redelivered
        )

        # ===== 步骤 2 & 3：验证参数并转换为 dataclass =====
        if msg_type == "model":
            # 验证头模生成参数
            validate_model_params(params_data)
            params = dict_to_model_params(params_data, task_id)
            task_handler = handle_model_task

        elif msg_type == "forward":
            # 验证正向仿真参数
            validate_forward_params(params_data)
            params = dict_to_forward_params(params_data, task_id)
            task_handler = handle_forward_task

        elif msg_type == "inverse":
            # 验证逆向仿真参数
            validate_inverse_params(params_data)
            params = dict_to_inverse_params(params_data, task_id)
            task_handler = handle_inverse_task

        elif msg_type == "ack_test":
            # 验证 ACK 测试参数
            validate_ack_test_params(params_data)
            params = dict_to_ack_test_params(params_data, task_id)
            task_handler = handle_ack_test_task

        else:
            # 未知任务类型，直接 ack 丢弃
            raise ValueError(f"未知任务类型: {msg_type}")

        # ===== 步骤 4：启动工作线程 =====
        worker = threading.Thread(
            target=execute_task,
            args=(
                message_queue,
                task_id,
                msg_type,
                task_handler,
                params,
                channel,
                delivery_tag,
                redelivered,
            ),
            daemon=True,  # daemon 线程会在主进程退出时自动终止
            name=f"task-{msg_type}-{task_id}",
        )
        worker.start()
        logger.info(
            "任务已提交到工作线程: %s, 类型: %s, 线程: %s",
            task_id,
            msg_type,
            worker.name,
        )

    except ValidationError as e:
        # 参数验证失败，记录日志并 ack（不重发，避免死循环）
        logger.error("参数验证失败: %s - %s", task_id, e)
        send_progress(message_queue, task_id, msg_type, 0, str(e))
        channel.basic_ack(delivery_tag=delivery_tag)

    except Exception as e:
        # 其他异常（JSON 解析失败等），记录日志并 ack
        logger.error("任务执行失败: %s - %s", task_id, e)
        send_progress(message_queue, task_id, msg_type, 0, str(e))
        channel.basic_ack(delivery_tag=delivery_tag)


def run_service(config: dict) -> None:
    """
    启动 RabbitMQ 服务

    服务架构
    ------
    本服务包含两个独立组件：

    1. RabbitMQPublisher（发送线程）
       - 独立线程，从 message_queue 队列中取消息
       - 将消息发送到后端的 simnibs_to_backend 队列
       - 用于发送进度更新和任务结果

    2. RabbitMQConsumer（监听线程/主线程）
       - 监听后端发送的 backend_to_simnibs 队列
       - 收到任务消息后触发 handle_message 回调
       - handle_message 启动工作线程处理任务

    消息流
    ------
    后端 -> backend_to_simnibs 队列 -> 监听线程 -> handle_message -> 工作线程
                                                                    |
                                                                    v
                                                      execute_task -> schedule_ack
                                                                    |
                                                                    v
                                          message_queue <- send_progress（进度更新）
                                                                    |
                                                                    v
                                          simnibs_to_backend 队列 <- 发送线程

    启动流程
    ------
    1. 创建共享的 Queue（message_queue）
    2. 启动发送线程（RabbitMQPublisher）
    3. 启动监听线程（RabbitMQConsumer），传入 handle_message 回调
    4. 监听线程阻塞运行，收到消息后回调处理
    5. 服务停止时，先停止监听，再停止发送线程

    Parameters
    ----------
    config : dict
        RabbitMQ 配置字典，包含以下键：
        - host: RabbitMQ 服务器地址
        - port: RabbitMQ 服务器端口
        - username: 用户名
        - password: 密码
        - virtual_host: 虚拟主机
        - heartbeat: 心跳间隔（秒）
        - blocked_connection_timeout: 阻塞连接超时（秒）
        - socket_timeout: socket 超时（秒）
        - connection_attempts: 连接重试次数
        - retry_delay: 重试间隔（秒）
        - send_queue_name: 发送队列名称（simnibs_to_backend）
        - listen_queue_name: 监听队列名称（backend_to_simnibs）
    """
    # 创建共享的消息队列，用于工作线程和发送线程之间的通信
    message_queue: Queue = Queue()

    # ===== 启动发送线程 =====
    # 发送线程负责将进度更新和任务结果发送回后端
    sender = RabbitMQPublisher(
        host=config["host"],
        port=config["port"],
        queue_name=config["send_queue_name"],
        message_queue=message_queue,
        username=config["username"],
        password=config["password"],
        virtual_host=config["virtual_host"],
        heartbeat=config["heartbeat"],
        blocked_connection_timeout=config["blocked_connection_timeout"],
        socket_timeout=config["socket_timeout"],
        connection_attempts=config["connection_attempts"],
        retry_delay=config["retry_delay"],
    )
    sender.start()  # 启动发送线程

    # ===== 启动监听线程（消费线程） =====
    # 监听线程负责接收后端发送的任务消息
    listener = RabbitMQConsumer(
        host=config["host"],
        port=config["port"],
        queue_name=config["listen_queue_name"],
        username=config["username"],
        password=config["password"],
        virtual_host=config["virtual_host"],
        heartbeat=config["heartbeat"],
        blocked_connection_timeout=config["blocked_connection_timeout"],
        socket_timeout=config["socket_timeout"],
        connection_attempts=config["connection_attempts"],
        retry_delay=config["retry_delay"],
    )

    logger.info("SimNIBS RabbitMQ 服务已启动")
    logger.info("监听队列: %s", config["listen_queue_name"])
    logger.info("发送队列: %s", config["send_queue_name"])

    # 使用 partial 绑定 message_queue 到回调函数
    callback = partial(handle_message, message_queue=message_queue)

    try:
        # 启动监听，进入阻塞循环，持续接收和处理消息
        listener.consume_forever(callback)
    finally:
        # 服务停止时，先停止发送线程
        sender.stop()
