# Hướng 7: Open-Vocabulary Traffic Scene Understanding via Delta Proposals

## 1. Giới thiệu & Đóng góp Khoa học (Novelty ⭐⭐⭐⭐)
Các phương pháp phát hiện phương tiện truyền thống bị giới hạn trong một tập danh mục cố định (Closed-Set Categories như COCO 80 lớp hoặc VOC 20 lớp). Khi triển khai thực tế trên đường phố Việt Nam, hệ thống không thể nhận diện các phương tiện đặc thù (xe ba gác, xe xích lô, xe cứu thương, xe chở rác) hoặc các chướng ngại vật bất thường trên đường.

**Đóng góp cốt lõi**:
1. **Delta Region Proposal Engine**: Sinh các hộp đề xuất vùng đối tượng không phụ thuộc lớp (Class-Agnostic Proposals) từ trường sai khác quang học $\Delta = \|I_{\text{origin}} - I_{\text{bg}}\|$ thay thế hoàn toàn mạng RPN hoặc Selective Search.
2. **DINO-to-CLIP Vision-Language Alignment**: Chiếu biểu diễn thị giác giàu chi tiết của DINO ViT sang không gian ngữ nghĩa văn bản CLIP (512 chiều).
3. **Zero-Shot Prompt Matching**: Cho phép người vận hành truy vấn và phát hiện bất kỳ đối tượng nào chỉ bằng mô tả văn bản tự nhiên (tiếng Việt hoặc tiếng Anh).

---

## 2. Cấu trúc Thư mục
- `proposal_engine.py`: `DeltaProposalEngine` sinh và lọc các hộp đề xuất vùng tiềm năng qua NMS.
- `text_prompts.py`: `TrafficPromptVocabulary` quản lý từ điển văn bản và prompt ngữ cảnh giao thông đô thị.
- `models.py`: `OpenVocabTrafficDetector` (ViT Backbone + Alignment Projector sang không gian CLIP).
- `pipeline.py`: Pipeline nhận diện hoàn chỉnh, vẽ bounding box và nhãn văn bản lên ảnh, xuất báo cáo CSV.

---

## 3. Hướng Dẫn Chạy (CLI Execution)

### Nhận diện với từ vựng tiếng Việt mặc định:
```bash
python direction7_open_vocabulary/pipeline.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --output_dir checkpoints/direction7_open_vocabulary \
    --backbone dinov3_vits16 \
    --language vi \
    --device cuda
```

### Nhận diện với danh mục tùy biến theo nhu cầu:
```bash
python direction7_open_vocabulary/pipeline.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --output_dir checkpoints/direction7_open_vocabulary/custom \
    --custom_classes "xe máy,ô tô con,xe cứu hỏa,xe cứu thương,người đi bộ" \
    --device cuda
```
