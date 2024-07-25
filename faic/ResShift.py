from omegaconf import OmegaConf
from faic.sampler import ResShiftSampler


def get_configs():
    configs = OmegaConf.load("./configs/faceir_gfpgan512_lpips.yaml")
    chop_stride = (512 - 64) * 4

    configs.model.ckpt_path = "models/resshift_faceir_s4.pth"
    configs.diffusion.params.sf = 1
    configs.autoencoder.ckpt_path = "models/ffhq512_vq_f8_dim8_face.pth"

    return configs, chop_stride


def load_model():
    configs, chop_stride = get_configs()

    sampler = ResShiftSampler(
        configs,
        sf=1,
        chop_size=2048,
        chop_stride=chop_stride,
        chop_bs=1,
        use_amp=True,
        seed=12345,
        padding_offset=configs.model.params.get("lq_size", 64),
    )

    return sampler
