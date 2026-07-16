# Satellite Image Matching

## Limitations

Current limitations of the proposed pipeline include:

fixed tile size (1024 × 1024);
independent processing of image tiles;
no geometric verification (e.g., RANSAC) after LoFTR matching;
no fine-tuning of the pretrained model on Sentinel-2 imagery.

Future improvements may include overlapping tiles, post-processing using
RANSAC, and training or fine-tuning LoFTR on remote sensing datasets.

## Future Improvements
Cloud masks supplied with Sentinel-2 products could be used
to exclude unreliable regions before keypoint detection,
improving matching robustness.

Повний Sentinel-2 TCI має розмір приблизно 10980×10980 пікселів. Через квадратичну складність attention у LoFTR це призводить до надмірного споживання пам'яті. Для демонстрації алгоритму було використано crop високої роздільної здатності, що зберігає локальні особливості сцени та дозволяє виконати matching без втрати якості.

Future work may include evaluating correspondence quality inside deforestation regions using the provided GeoJSON annotations.

Due to memory limitations of the available GPU environment, image matching was performed on high-resolution central crops instead of the complete Sentinel-2 scenes
