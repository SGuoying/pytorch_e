# %%
import torch
from composer.loss import soft_cross_entropy
from composer.metrics import CrossEntropy
from composer.models import ComposerClassifier
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from sunyata.pytorch.arch.vision_transformer import ViT, ViTCfg, bayes_ViT


# %%
def build_composer_convmixer(model_name: str = 'vit',
                             num_layers: int = 8,
                             hidden_dim: int = 256,
                             image_size: int = 224,
                             patch_size: int = 32,
                             num_heads: int = 8,
                             num_classes: int = 100,
                             expanded_dim: int = 512,
                             head_dim: int = 32,
                             ):
    
    cfg = ViTCfg(
        num_layers = num_layers,
        hidden_dim = hidden_dim,
        image_size = image_size,
        patch_size = patch_size,
        num_heads = num_heads,
        num_classes = num_classes,
        expanded_dim = expanded_dim,
        head_dim = head_dim,
    )

    if model_name == "vit":
        model = ViT(cfg)
    elif model_name == "bayesvit":
        model = bayes_ViT(cfg)

    else:
        raise ValueError(f"model_name='{model_name}' but only 'convmixer' and 'bayes_convmixer' are supported now.")
    
    # Performance metrics to log other than training loss
    train_metrics = MulticlassAccuracy(num_classes=num_classes, average='micro')
    val_metrics = MetricCollection([
        CrossEntropy(),
        MulticlassAccuracy(num_classes=num_classes, average='micro')
    ])

    # Wrapper function to convert a image classification Pytorch model into a Composer model
    composer_model = ComposerClassifier(
        model,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        loss_fn=soft_cross_entropy,
    )
    return composer_model
# %%
# composer_model = build_composer_convmixer()
# input = [torch.randn(2, 3, 224, 224), torch.randint(0,100, (2,))]
# output = composer_model(input)
# output.shape
