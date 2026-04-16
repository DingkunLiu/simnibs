from __future__ import division
from enum import IntEnum


class ElementTypes(IntEnum):
    LINE = 1
    TRIANGLE = 2
    TETRAHEDRON = 4
    POINT = 15


class ElementTags(IntEnum):
    @classmethod
    def from_surface_file_name(cls, surface_name: str, hemisphere: str):
        tag_from_name = {
            ("gray", "lh"): cls.LH_PIAL_SURFACE,
            ("gray", "rh"): cls.RH_PIAL_SURFACE,
            ("pial", "lh"): cls.LH_PIAL_SURFACE,
            ("pial", "rh"): cls.RH_PIAL_SURFACE,
            ("white", "lh"): cls.LH_WM_SURFACE,
            ("white", "rh"): cls.RH_WM_SURFACE,
            ("sphere", "lh"): cls.LH_SPHERE,
            ("sphere", "rh"): cls.RH_SPHERE,
            ("sphere.reg", "lh"): cls.LH_SPHERE_REG,
            ("sphere.reg", "rh"): cls.RH_SPHERE_REG,
            ("central", "lh"): cls.LH_CENTRAL_SURFACE,
            ("central", "rh"): cls.RH_CENTRAL_SURFACE,
        }
        return tag_from_name[(surface_name.lower(), hemisphere.lower())]

    TH_START = 0
    WM = 1
    GM = 2
    CSF = 3
    BONE = 4
    SCALP = 5
    EYE_BALLS = 6
    COMPACT_BONE = 7
    SPONGY_BONE = 8
    BLOOD = 9
    MUSCLE = 10
    CARTILAGE = 11
    FAT = 12
    ELECTRODE_RUBBER_START = 100
    ELECTRODE_RUBBER = 100
    ELECTRODE_RUBBER_END = 499
    SALINE_START = 500
    SALINE = 500
    SALINE_END = 899
    CREAM = 999
    TH_END = 999

    TH_SURFACE_START = 1000
    WM_TH_SURFACE = TH_SURFACE_START + WM
    GM_TH_SURFACE = TH_SURFACE_START + GM
    CSF_TH_SURFACE = TH_SURFACE_START + CSF
    BONE_TH_SURFACE = TH_SURFACE_START + BONE
    SCALP_TH_SURFACE = TH_SURFACE_START + SCALP
    EYE_BALLS_TH_SURFACE = TH_SURFACE_START + EYE_BALLS
    COMPACT_BONE_TH_SURFACE = TH_SURFACE_START + COMPACT_BONE
    SPONGY_BONE_TH_SURFACE = TH_SURFACE_START + SPONGY_BONE
    BLOOD_TH_SURFACE = TH_SURFACE_START + BLOOD
    MUSCLE_TH_SURFACE = TH_SURFACE_START + MUSCLE
    CARTILAGE_TH_SURFACE = TH_SURFACE_START + CARTILAGE
    FAT_TH_SURFACE = TH_SURFACE_START + FAT
    INTERNAL_AIR_TH_SURFACE = TH_SURFACE_START + ELECTRODE_RUBBER_START - 1

    ELECTRODE_RUBBER_TH_SURFACE_START = TH_SURFACE_START + ELECTRODE_RUBBER_START
    ELECTRODE_RUBBER_TH_SURFACE = TH_SURFACE_START + ELECTRODE_RUBBER
    ELECTRODE_RUBBER_TH_SURFACE_END = TH_SURFACE_START + ELECTRODE_RUBBER_END
    SALINE_TH_SURFACE_START = TH_SURFACE_START + SALINE_START
    SALINE_TH_SURFACE = TH_SURFACE_START + SALINE
    SALINE_TH_SURFACE_END = TH_SURFACE_START + SALINE_END

    ELECTRODE_PLUG_SURFACE_START = 2000
    ELECTRODE_PLUG_SURFACE = ELECTRODE_PLUG_SURFACE_START + ELECTRODE_RUBBER_START
    ELECTRODE_PLUG_SURFACE_END = 2499
    TH_SURFACE_END = 2499

    LH_SURFACE_START = 5000
    LH_WM_SURFACE = LH_SURFACE_START + WM
    LH_PIAL_SURFACE = LH_SURFACE_START + GM
    LH_CENTRAL_SURFACE = LH_SURFACE_START + GM + 1
    LH_SPHERE = LH_SURFACE_START + 100
    LH_SPHERE_REG = LH_SURFACE_START + 101
    LH_SURFACE_END = 5499

    LH_LAYER_START = 5500
    LH_LAYER_1 = LH_LAYER_START + 100
    LH_LAYER_2_3 = LH_LAYER_START + 101
    LH_LAYER_4 = LH_LAYER_START + 102
    LH_LAYER_5 = LH_LAYER_START + 103
    LH_LAYER_6 = LH_LAYER_START + 104
    LH_BORDER_1_2 = LH_LAYER_START + 200
    LH_BORDER_2_3 = LH_LAYER_START + 201
    LH_BORDER_3_4 = LH_LAYER_START + 202
    LH_BORDER_5_6 = LH_LAYER_START + 203
    LH_BORDER_6_WM = LH_LAYER_START + 204
    LH_LAYER_END = 5999

    RH_SURFACE_START = 7000
    RH_WM_SURFACE = RH_SURFACE_START + WM
    RH_PIAL_SURFACE = RH_SURFACE_START + GM
    RH_CENTRAL_SURFACE = RH_SURFACE_START + GM + 1
    RH_SPHERE = RH_SURFACE_START + 100
    RH_SPHERE_REG = RH_SURFACE_START + 101
    RH_SURFACE_END = 7499

    RH_LAYER_START = 7500
    RH_LAYER_1 = RH_LAYER_START + 100
    RH_LAYER_2_3 = RH_LAYER_START + 101
    RH_LAYER_4 = RH_LAYER_START + 102
    RH_LAYER_5 = RH_LAYER_START + 103
    RH_LAYER_6 = RH_LAYER_START + 104
    RH_BORDER_1_2 = RH_LAYER_START + 200
    RH_BORDER_2_3 = RH_LAYER_START + 201
    RH_BORDER_3_4 = RH_LAYER_START + 202
    RH_BORDER_5_6 = RH_LAYER_START + 203
    RH_BORDER_6_WM = RH_LAYER_START + 204
    RH_LAYER_END = 7999

    UNKNOWN_SURFACE = 9999


class CentralLayerDepths:
    CENTRAL_LAYER_1 = 0.06
    CENTRAL_LAYER_23 = 0.4
    CENTRAL_LAYER_4 = 0.55
    CENTRAL_LAYER_5 = 0.65
    CENTRAL_LAYER_6 = 0.85


central_cortical_layer_depths: dict[int, float] = {
    ElementTags.LH_LAYER_1: CentralLayerDepths.CENTRAL_LAYER_1,
    ElementTags.RH_LAYER_1: CentralLayerDepths.CENTRAL_LAYER_1,
    ElementTags.LH_LAYER_2_3: CentralLayerDepths.CENTRAL_LAYER_23,
    ElementTags.RH_LAYER_2_3: CentralLayerDepths.CENTRAL_LAYER_23,
    ElementTags.LH_LAYER_4: CentralLayerDepths.CENTRAL_LAYER_4,
    ElementTags.RH_LAYER_4: CentralLayerDepths.CENTRAL_LAYER_4,
    ElementTags.LH_LAYER_5: CentralLayerDepths.CENTRAL_LAYER_5,
    ElementTags.RH_LAYER_5: CentralLayerDepths.CENTRAL_LAYER_5,
    ElementTags.LH_LAYER_6: CentralLayerDepths.CENTRAL_LAYER_6,
    ElementTags.RH_LAYER_6: CentralLayerDepths.CENTRAL_LAYER_6,
}

central_cortical_layer_names: dict[int, str] = {
    ElementTags.LH_LAYER_1: "central_cl_1",
    ElementTags.RH_LAYER_1: "central_cl_1",
    ElementTags.LH_LAYER_2_3: "central_cl_23",
    ElementTags.RH_LAYER_2_3: "central_cl_23",
    ElementTags.LH_LAYER_4: "central_cl_4",
    ElementTags.RH_LAYER_4: "central_cl_4",
    ElementTags.LH_LAYER_5: "central_cl_5",
    ElementTags.RH_LAYER_5: "central_cl_5",
    ElementTags.LH_LAYER_6: "central_cl_6",
    ElementTags.RH_LAYER_6: "central_cl_6",
}

central_cortical_layer_tags: list[int] = [
    ElementTags.LH_LAYER_1,
    ElementTags.RH_LAYER_1,
    ElementTags.LH_LAYER_2_3,
    ElementTags.RH_LAYER_2_3,
    ElementTags.LH_LAYER_4,
    ElementTags.RH_LAYER_4,
    ElementTags.LH_LAYER_5,
    ElementTags.RH_LAYER_5,
    ElementTags.LH_LAYER_6,
    ElementTags.RH_LAYER_6,
]

tissue_tags: list[int] = [
    ElementTags.WM,
    ElementTags.GM,
    ElementTags.CSF,
    ElementTags.BONE,
    ElementTags.SCALP,
    ElementTags.EYE_BALLS,
    ElementTags.COMPACT_BONE,
    ElementTags.SPONGY_BONE,
    ElementTags.BLOOD,
    ElementTags.MUSCLE,
    ElementTags.CARTILAGE,
    ElementTags.FAT,
    ElementTags.ELECTRODE_RUBBER,
    ElementTags.SALINE,
]

tissue_names: dict[int, str] = {
    ElementTags.WM: "WM",
    ElementTags.GM: "GM",
    ElementTags.CSF: "CSF",
    ElementTags.BONE: "Bone",
    ElementTags.SCALP: "Scalp",
    ElementTags.EYE_BALLS: "Eye_balls",
    ElementTags.COMPACT_BONE: "Compact_bone",
    ElementTags.SPONGY_BONE: "Spongy_bone",
    ElementTags.BLOOD: "Blood",
    ElementTags.MUSCLE: "Muscle",
    ElementTags.CARTILAGE: "Cartilage",
    ElementTags.FAT: "Fat",
    ElementTags.ELECTRODE_RUBBER: "Electrode_rubber",
    ElementTags.SALINE: "Saline",
}

tissue_conductivities: dict[int, float] = {
    ElementTags.WM: 0.126,
    ElementTags.GM: 0.275,
    ElementTags.CSF: 1.654,
    ElementTags.BONE: 0.010,
    ElementTags.SCALP: 0.465,
    ElementTags.EYE_BALLS: 0.5,
    ElementTags.COMPACT_BONE: 0.008,
    ElementTags.SPONGY_BONE: 0.025,
    ElementTags.BLOOD: 0.6,
    ElementTags.MUSCLE: 0.16,
    ElementTags.CARTILAGE: 0.88,
    ElementTags.FAT: 0.078,
    ElementTags.ELECTRODE_RUBBER: 29.4,
    ElementTags.SALINE: 1.0,
}

tissue_conductivity_descriptions: dict[int, str] = {
    ElementTags.WM: "brain white matter (from Wagner 2004)",
    ElementTags.GM: "brain gray matter (from Wagner 2004)",
    ElementTags.CSF: "cerebrospinal fluid (from Wagner 2004)",
    ElementTags.BONE: "average bone (from Wagner 2004)",
    ElementTags.SCALP: "average scalp (from Wagner 2004)",
    ElementTags.EYE_BALLS: "vitreous humour (from Opitz, Paulus, Thielscher, submitted)",
    ElementTags.COMPACT_BONE: "compact bone (from Opitz, Paulus, Thielscher, submitted)",
    ElementTags.SPONGY_BONE: "spongy bone (from Opitz, Paulus, Thielscher, submitted)",
    ElementTags.BLOOD: "Blood (from Gabriel et al, 2009)",
    ElementTags.MUSCLE: "Muscle (from Gabriel et al, 2009)",
    ElementTags.CARTILAGE: "Cartilage (average of values from Binette et al. 2004 and Morita et al. 2012)",
    ElementTags.FAT: "Fat (from Gabriel et al, 2009)",
    ElementTags.ELECTRODE_RUBBER: "for tDCS rubber electrodes",
    ElementTags.SALINE: "for tDCS sponge electrodes",
}
