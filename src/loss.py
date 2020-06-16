import torch


def wsdr_loss(pred_y_batch,
              y_with_noise_batch,
              y_batch,
              noise_batch,
              reduction='sum'):
    data = zip(*[
        x.unbind(0) for x in [
            pred_y_batch,
            y_with_noise_batch,
            y_batch,
            noise_batch,
        ]
    ])
    result = 0
    for (pred_y, y_with_noise, y, noise) in data:

        pred_noise = y_with_noise - pred_y
        pred_noise /= max(
            abs(pred_noise.min()),
            abs(pred_noise.max()),
            1e-12,
        )

        y_norm = torch.norm(y, 2)
        noise_norm = torch.norm(noise, 2)
        pred_y_norm = torch.norm(pred_y, 2)
        pred_noise_norm = torch.norm(pred_noise, 2)

        prod_y = torch.sqrt(torch.sum((pred_y - y)**2))
        sdr_y_loss = prod_y / y_norm / pred_y_norm

        prod_noise = torch.sqrt(torch.sum((pred_noise - noise)**2))
        sdr_noise_loss = prod_noise / noise_norm / pred_noise_norm

        alpha = y_norm**2 / (noise_norm**2 + y_norm**2)

        loss = alpha * sdr_y_loss + (1 - alpha) * sdr_noise_loss

        result += loss

    if reduction == 'mean':
        result = result / y_batch.shape[0]
    elif reduction == 'sum':
        pass
    else:
        raise ValueError

    return result
