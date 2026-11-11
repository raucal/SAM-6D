"""Shared explicit device/options setup for the two supported proposal generators."""
def configure(cfg, segmentor='sam', backbone='vit_h', stability=0.97):
    from settings import checkpoint
    from omegaconf import OmegaConf
    OmegaConf.update(cfg, 'model.descriptor_model.checkpoint_path', str(checkpoint('dinov2')), force_add=True)
    if segmentor == 'fastsam':
        cfg.model.segmentor_model.checkpoint_path = str(checkpoint('fastsam'))
    if segmentor == 'sam':
        OmegaConf.update(cfg, 'model.segmentor_model.sam.checkpoint_path', str(checkpoint('sam', backbone)), force_add=True)
        cfg.model.segmentor_model.sam.model_type = backbone
        cfg.model.segmentor_model.stability_score_thresh = stability

def move_segmentor(model, device, segmentor='sam'):
    if segmentor == 'sam':
        model.segmentor_model.predictor.model.to(device)
    else:
        model.segmentor_model.current_device = device
