"""Shared explicit device/options setup for the two supported proposal generators."""
def configure(cfg, segmentor='sam', backbone='vit_h', stability=0.97):
    if segmentor == 'sam':
        cfg.model.segmentor_model.sam.model_type = backbone
        cfg.model.segmentor_model.stability_score_thresh = stability

def move_segmentor(model, device, segmentor='sam'):
    if segmentor == 'sam':
        model.segmentor_model.predictor.model.to(device)
    else:
        model.segmentor_model.current_device = device
