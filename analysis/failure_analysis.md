# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Đức Anh  
**Mã số học viên:** 2A202602888  
**Khóa:** K4 - Track 3B  

---

## 1. RAGAS Scores Comparison

| Metric | Naive Baseline | Production Pipeline | Δ (Cải thiện) | Đánh giá |
|--------|:--------------:|:-------------------:|:-------------:|:--------:|
| **Faithfulness** | 0.7667 | **0.8742** | **+0.1075** | ✅ Vượt mục tiêu bonus (≥ 0.85) |
| **Answer Relevancy** | 0.7672 | **0.8401** | **+0.0730** | ✅ Đạt chuẩn cao (≥ 0.75) |
| **Context Precision** | 0.9250 | **0.9500** | **+0.0250** | ✅ Top-k chính xác cao |
| **Context Recall** | 0.8833 | **0.9250** | **+0.0417** | ✅ Phủ đầy đủ thông tin |

> **Nhận xét tổng quan:**  
> - Toàn bộ 4 metrics trong Production RAG Pipeline đều vượt mốc **0.84** (vượt xa yêu cầu chuẩn 0.75).
> - Nhờ áp dụng **Parent-Child Hierarchical Retrieval**, các bảng biểu và ngữ cảnh tài liệu không bị cắt vụn như trong Basic Chunking.
> - Kỹ thuật **Enrichment (Contextual Prepend + Metadata)** và **Hybrid Search (BM25 + Dense BGE-M3 + RRF)** kết hợp **Cross-Encoder Reranker** giúp độ chính xác của ngữ cảnh tăng vọt lên 0.9500 và khả năng bao quát thông tin đạt 0.9250.

---

## 2. Latency Breakdown Report (Bonus Criterion)

Bảng phân tích độ trễ qua từng công đoạn xử lý trong pipeline thực tế:

| Thành phần Pipeline | Kỹ thuật áp dụng | Thời gian trung bình / Run | Tỷ lệ thời gian | Nhận xét & Đánh giá |
|---------------------|------------------|---------------------------:|:---------------:|---------------------|
| **M1: Document Chunking** | Hierarchical (Parent 2048 / Child 256) | 0.1s | 0.03% | Cực nhanh, chia 26 docs thành 119 child chunks |
| **M5: Chunk Enrichment** | Combined single-call (ThreadPool 8 workers) | 48.9s | 14.2% | ~0.41s/chunk qua GPT-4o-mini |
| **M2: Hybrid Indexing** | BM25 + Qdrant Dense BGE-M3 | 25.6s | 7.4% | Dense embedding 119 chunks tốn phần lớn thời gian |
| **M3: Model Loading & Rerank** | BAAI/bge-reranker-v2-m3 | 0.0s (cached) / ~180ms/query | 1.1% | Cross-encoder load 1 lần, rerank top-20 → top-3 cực nhanh |
| **Pipeline Retrieval & Generation** | Top-3 Rerank + GPT-4o-mini generation | ~2.5s / query | 14.5% | Tạo câu trả lời có trích dẫn context |
| **M4: RAGAS Evaluation** | 4 metrics × 20 questions (80 eval calls) | 32.8s | 9.5% | Đánh giá độc lập bằng LLM Judge |

---

## 3. Bottom-5 Failures Analysis

### #1. Tính phí phạt chậm thanh toán tạm ứng
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Phí phạt 5 ngày: 15.000.000 VNĐ * (2%/30 ngày) * 5 ngày = 50.025 VNĐ.
- **Worst metric:** Faithfulness (0.4000)
- **Error Tree:** Output đúng bản chất logic → Context đúng? Có (chính sách tạm ứng nêu rõ 15 ngày và 2%/tháng) → Query OK? OK → LLM reasoning / math assumption.
- **Root cause:** Context gốc ghi phạt "2%/tháng trên số tiền chưa hoàn ứng" mà không ghi công thức chia ngày. LLM tự động suy luận chia cho 30 ngày (tính ra 0.0667%/ngày dẫn đến số lẻ 50.025 VNĐ). RAGAS trích xuất claim và đánh giá việc chia 30 ngày là claim không có trong context gốc.
- **Suggested fix:** Cập nhật system prompt: *"Đối với các bài toán tính toán, trích dẫn công thức nguyên văn từ chính sách trước khi tính, nếu văn bản không quy định số ngày chuẩn trong tháng thì nêu rõ giả định trước khi tính"*.

---

### #2. Thẩm quyền duyệt nghỉ phép không lương 20 ngày
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần phê duyệt của Giám đốc điều hành (CEO). Lưu ý: nghỉ trên 14 ngày không lương, nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** Nghỉ phép không lương 20 ngày cần phê duyệt của Giám đốc Nhân sự.
- **Worst metric:** Faithfulness (0.0000)
- **Error Tree:** Output sai mốc phê duyệt → Context đúng? Có đầy đủ 3 mốc (1-5 ngày: Trưởng phòng; 6-15 ngày: GĐ Nhân sự; 16-30 ngày: CEO) → LLM matching error.
- **Root cause:** Context chứa nhiều khoảng giá trị liên tiếp gần nhau. LLM bị nhầm mốc 20 ngày với mốc 6-15 ngày (Giám đốc Nhân sự).
- **Suggested fix:** Thêm chỉ dẫn Chain-of-Thought (CoT) vào prompt: *"Trước khi kết luận người phê duyệt, hãy xác định câu hỏi rơi vào khoảng số ngày nào trong quy chế"*.

---

### #3. Hoàn trả chi phí đào tạo khi nghỉ việc trước hạn
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Got:** Nhân viên đã được tài trợ 25 triệu VNĐ và cam kết làm việc ít nhất 1 năm... nghỉ sau 8 tháng là chưa đủ thời gian... phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Worst metric:** Faithfulness (0.3333)
- **Error Tree:** Output hoàn toàn đúng đáp án và logic nghiệp vụ → Context đúng? Đúng → RAGAS Claim Extraction discrepancy.
- **Root cause:** LLM diễn đạt dài dòng, bổ sung các câu lập luận diễn dịch ("tức là chưa đủ thời gian cam kết"). RAGAS phân rã câu thành các mệnh đề nhỏ và đánh giá một số mệnh đề diễn giải là "ungrounded".
- **Suggested fix:** Bổ sung ràng buộc trả lời ngắn gọn: *"Trả lời súc tích, trực tiếp, chỉ dùng các câu khẳng định bám sát dữ liệu trong tài liệu, tránh các câu suy luận mang tính chất cá nhân"*.

---

### #4. Điều kiện thâm niên cộng ngày phép năm (Xung đột phiên bản)
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got:** Nhân viên có thâm niên từ 3 năm trở lên sẽ được cộng thêm 1 ngày phép cho mỗi 3 năm làm việc liên tục theo chính sách nghỉ phép năm (Phiên bản 2024).
- **Worst metric:** Context Precision (0.5000)
- **Error Tree:** Output đúng quy định mới nhất → Context có cả 2 phiên bản v2023 và v2024 → RAGAS đánh giá chunk v2023 là irrelevant.
- **Root cause:** Do cả 2 tài liệu v2023 và v2024 đều bàn về cùng chủ đề và dùng từ khóa tương tự, BM25 và Dense Search đều kéo cả 2 file về top kết quả. RAGAS xem tài liệu cũ là nhiễu, làm giảm điểm Precision.
- **Suggested fix:** Sử dụng Metadata Filtering ở M2: lọc `status != 'superseded'` hoặc `version == 'latest'` ngay trong câu truy vấn Qdrant vector search.

---

### #5. Số ngày nghỉ phép năm tiêu chuẩn
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got:** Nhân viên chính thức được hưởng 15 ngày phép năm có lương theo chính sách nghỉ phép năm (Phiên bản 2024), tăng từ 12 ngày so với chính sách năm 2023.
- **Worst metric:** Context Precision (0.5000)
- **Error Tree:** Output hoàn hảo → Context có cả văn bản v2023 (12 ngày) và v2024 (15 ngày) → Context Precision bị kéo xuống 0.5.
- **Root cause:** Tương tự câu #4, hiện tượng document versioning không được lọc từ khâu retrieval mà chỉ được giải quyết ở tầng LLM prompt reasoning.
- **Suggested fix:** Tích hợp Metadata Filtering hoặc Temporal Re-ranking (ưu tiên văn bản có ngày hiệu lực mới hơn).

---

## 4. Case Study chuyên sâu (Dành cho Presentation)

**Câu hỏi phân tích:**  
> *"Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?"*  
> *(Ground truth: Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt).*

### Error Tree Walkthrough:
1. **Kiểm tra Output:** Ở phiên bản ban đầu chưa có Parent Expansion, LLM trả về *"Không tìm thấy."* (Sai hoàn toàn).
2. **Kiểm tra Context:** 
   - Chunk Markdown table thẩm quyền phê duyệt bị cắt ngang ở ngưỡng 256 ký tự:  
     `| Trên **50.000.000 VNĐ** |` (cột người phê duyệt `Tổng Giám đốc (CEO) |` bị rơi sang chunk kế tiếp).
   - Do đó, dù retrieval tìm đúng chunk bảng mua sắm, thông tin cần thiết lại bị cụt.
3. **Kiểm tra Query & Retrieval:** Query tìm kiếm và Reranking hoạt động tốt (tìm trúng file `mua_sam.md`).
4. **Điểm cần Fix:** Bước **M1 Chunking & Context Expansion**.

### Giải pháp khắc phục đã triển khai:
- Áp dụng triệt để mô hình **Parent-Child Hierarchical Retrieval**: Dùng child chunk (256 ký tự) để đạt độ chính xác tìm kiếm cao nhất (high precision), nhưng khi tổng hợp context cho LLM thì truy ngược lại `parent_id` để lấy toàn bộ Parent chunk (2048 ký tự).
- Kết quả sau khi fix: Bảng thẩm quyền được giữ trọn vẹn, LLM trả lời chính xác 100% là *"Tổng Giám đốc (CEO) phê duyệt"*, nâng điểm Faithfulness và Relevancy lên mức xuất sắc.

---

## 5. Kế hoạch tối ưu hóa nếu có thêm 1 giờ
1. **Metadata Filtering tự động:** Đưa trường `version` và `is_active` vào Qdrant payload index để tự động loại bỏ các tài liệu hết hiệu lực khi truy vấn chính sách hiện hành.
2. **Self-Correction LLM Judge:** Tích hợp một bước kiểm tra tự động trước khi trả lời: so sánh câu trả lời với context, nếu phát hiện tính toán số học thì chạy code python sandbox để tính chính xác tuyệt đối.
3. **Sentence Window Retrieval:** Thêm kỹ thuật Sentence-Window để mở rộng ngữ cảnh 2 câu trước và 2 câu sau của câu match, giúp tối ưu hóa dung lượng token nạp vào prompt.
