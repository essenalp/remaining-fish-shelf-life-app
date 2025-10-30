# Balık Raf Ömrü Tahmin Uygulaması 🐟

Bu proje, balığın depolama koşullarına ve hasattan itibaren geçen süreye göre tahmini kalan raf ömrünü hesaplayan bir demo uygulamadır. Hasattan itibaren geçen gün sayısı; mevcut depolama şartlarını, hasat sonrası ortalama depolama sıcaklığı; mevcut depolama sıcaklığını, depolama süresi; soğuk zincir kırıldığı andan itibaren geçen süreyi (ihlal süresi) ve depolama sıcaklığı; 0 C'den sapmaları belirtir. 

Uygulama, **Random Forest** ve **XGBoost** modellerini kullanmaktadır ve kullanıcıya tek sayfada gerekli tüm parametreleri girme imkanı sunar.

## Özellikler

- Balık Türü seçimi (Somon / Levrek)
- Depolama süresi (saat)
- Depolama sıcaklığı (°C)
- Hasattan itibaren geçen gün sayısı
- Hasat sonrası ortalama depolama sıcaklığı (°C)
- Model Seçimi (Random Forest / XGBoost)
- Tahmini kalan raf ömrünü görsel olarak gösterir
