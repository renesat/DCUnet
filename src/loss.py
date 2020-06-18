import torch


def wsdr_loss(pred_y, y_with_noise, y, reduction='mean', eps=1e-8):
    pred_noise = y_with_noise - pred_y
    noise = y_with_noise - y

    y_norm = y.norm(2, dim=1)
    noise_norm = noise.norm(2, dim=1)
    pred_y_norm = pred_y.norm(2, dim=1)
    pred_noise_norm = pred_noise.norm(2, dim=1)

    prod_y = torch.sum(pred_y * y, dim=1)
    sdr_y_loss = -prod_y / (y_norm * pred_y_norm + eps)

    prod_noise = torch.sum(pred_noise * noise, dim=1)
    sdr_noise_loss = -prod_noise / (noise_norm * pred_noise_norm + eps)

    alpha = y_norm**2 / (noise_norm**2 + y_norm**2 + eps)

    loss = alpha * sdr_y_loss + (1 - alpha) * sdr_noise_loss

    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        raise ValueError
