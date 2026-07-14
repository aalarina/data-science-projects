# Satellite Image Matching

## Future Improvements
Cloud masks supplied with Sentinel-2 products could be used
to exclude unreliable regions before keypoint detection,
improving matching robustness.

Повний Sentinel-2 TCI має розмір приблизно 10980×10980 пікселів. Через квадратичну складність attention у LoFTR це призводить до надмірного споживання пам'яті. Для демонстрації алгоритму було використано crop високої роздільної здатності, що зберігає локальні особливості сцени та дозволяє виконати matching без втрати якості.

Future work may include evaluating correspondence quality inside deforestation regions using the provided GeoJSON annotations.

Due to memory limitations of the available GPU environment, image matching was performed on high-resolution central crops instead of the complete Sentinel-2 scenes
