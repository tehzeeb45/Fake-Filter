export const MODEL_LABELS = {
  new_xception: "XceptionNet",
  new_efficientnet: "EfficientNet-B3",
  new_vit_small: "ViT-Small",
  new_vit_large_clip: "ViT-Large/CLIP",
  cnn: "XceptionNet",
  efficientnet: "EfficientNet-B3",
  vit: "ViT-Small",
  vit_l14: "ViT-Large/CLIP",
};

export function formatModelScores(scores = {}) {
  return Object.entries(scores)
    .map(([k, v]) => (MODEL_LABELS[k] || k) + ": " + (+v * 100).toFixed(1) + "%")
    .join(" · ");
}