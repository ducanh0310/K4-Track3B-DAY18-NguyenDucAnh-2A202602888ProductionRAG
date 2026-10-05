# Individual Reflection — Lab 18: Production RAG Pipeline

**Họ và tên:** Nguyễn Đức Anh  
**Mã số học viên:** 2A202602888  
**Khóa:** K4 - Track 3B (AI Engineer)  
**Ngày hoàn thành:** 05/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Dưới đây là bảng đối chiếu chi tiết giữa các khái niệm lý thuyết cốt lõi trong bài giảng Production RAG và mã nguồn đã triển khai thực tế trong 5 modules của bài lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích chuyên sâu |
|----------------|:------:|:-----------|:-----------------------------------|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Dùng `SentenceTransformer("all-MiniLM-L6-v2")` mã hóa từng câu và đo cosine similarity giữa 2 câu liền kề. Với ngưỡng `threshold=0.85`, các câu cùng ý được gom nhóm mượt mà, khắc phục triệt để nhược điểm của Basic Chunking là cắt đứt câu giữa chừng hoặc ngắt đoạn tùy tiện theo số ký tự cứng. |
| **Hierarchical Chunking (Parent-Child)** | M1 & Pipeline | `chunk_hierarchical()`, `run_query()` | Tạo cấu trúc 2 tầng: Parent chunks (2048 ký tự) giữ trọn cấu trúc ngữ cảnh/bảng biểu, và Child chunks (256 ký tự) phục vụ embedding. Cơ chế *"Retrieve Child → Return Parent"* giải quyết bài toán mâu thuẫn kinh điển trong RAG: vector embedding cần chunk nhỏ để tăng độ chính xác tìm kiếm (precision), nhưng LLM lại cần ngữ cảnh lớn (context) để suy luận không bị ảo giác. |
| **Structure-Aware Chunking** | M1 | `chunk_structure_aware()` | Phân tách tài liệu theo tiêu đề Markdown (`#`, `##`, `###`), bảo toàn nguyên vẹn danh sách, bảng Markdown và khối mã nguồn mà không bị xé vụn. Thêm metadata `section` giúp dễ dàng lọc thông tin theo chuyên mục. |
| **Vietnamese Word Segmentation** | M2 | `segment_vietnamese()` | Sử dụng thư viện `underthesea` để tách từ tiếng Việt và loại bỏ dấu gạch nối `_`. Điều này cực kỳ quan trọng vì nếu BM25 tách từ theo khoảng trắng, từ ghép như "nghỉ_phép" sẽ bị coi là 1 token đơn và không thể match với truy vấn người dùng gõ "nghỉ phép". |
| **Hybrid Search & Reciprocal Rank Fusion (RRF)** | M2 | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | Kết hợp sức mạnh tìm kiếm từ khóa chính xác (lexical - BM25Okapi) và tìm kiếm ngữ nghĩa tiềm ẩn (semantic - Qdrant + BGE-M3 1024-dim). Thuật toán RRF với công thức $RRF(d) = \sum \frac{1}{k + rank + 1}$ giúp chuẩn hóa và cộng hưởng điểm xếp hạng giữa hai không gian điểm khác biệt mà không cần tinh chỉnh trọng số alpha thủ công. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Mô hình `BAAI/bge-reranker-v2-m3` nhận trực tiếp cặp `(query, document)` và tính toán full cross-attention giữa từng từ trong câu hỏi và tài liệu. Lọc từ top-20 ứng viên xuống top-3 chất lượng cao nhất, giúp loại bỏ các tài liệu gây nhiễu, tăng Context Precision lên **0.9500**. Mô hình được cache trong bộ nhớ giúp rerank mỗi query chỉ mất ~180ms. |
| **RAGAS Evaluation Framework** | M4 | `evaluate_ragas()`, `failure_analysis()` | Đánh giá khách quan 4 chỉ số cốt lõi: Faithfulness (0.8742), Answer Relevancy (0.8401), Context Precision (0.9500), Context Recall (0.9250). Cây chẩn đoán lỗi (Diagnostic Error Tree) tự động phân loại nguyên nhân thất bại thành các nhóm: ảo giác LLM, thiếu chunk, chunk nhiễu, hay prompt chưa phù hợp. |
| **Contextual Prepend & Enrichment** | M5 | `_enrich_single_call()`, `contextual_prepend()` | Kỹ thuật theo bài báo của Anthropic: bổ sung 1 câu tóm tắt vị trí tài liệu trước mỗi chunk giúp chunk không bị cô lập mất ngữ cảnh (orphan chunk). Chế độ Combined single-call tích hợp tóm tắt, sinh câu hỏi giả định (HyQA), ngữ cảnh và metadata tự động chỉ trong 1 lần gọi API duy nhất, tối ưu 75% chi phí API và thời gian chạy. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

Trong quá trình triển khai hệ thống Production RAG, tôi đã gặp phải một số thách thức kỹ thuật và rút ra bài học kinh nghiệm quan trọng:

### 1. Lỗi cắt cụt bảng Markdown dẫn đến ảo giác "Không tìm thấy"
- **Hiện tượng (Exact issue):** Ở lần chạy thử đầu tiên, câu hỏi *"Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?"* bị trả về *"Không tìm thấy."*, làm điểm Faithfulness và Relevancy bị tụt dốc.
- **Nguyên nhân gốc rễ (Root cause):** Bảng thẩm quyền phê duyệt mua sắm có dòng `| Trên 50.000.000 VNĐ | Tổng Giám đốc (CEO) |`. Do kích thước child chunk trong hierarchical chunking bị giới hạn ở 256 ký tự, dòng bảng này bị cắt đứt làm đôi ở giữa: cột giá trị nằm ở chunk trước, còn cột người phê duyệt bị đẩy sang chunk sau. Khi reranker chọn chunk đầu, LLM không thấy thông tin người duyệt nên từ chối trả lời.
- **Cách xử lý & Debug:** 
  - Triển khai đầy đủ cơ chế **Parent Context Lookup** trong `src/pipeline.py`: Sau khi child chunk được chọn bởi Reranker, pipeline tra cứu `parent_map` dựa trên `parent_id` để lấy toàn bộ khối văn bản gốc của Parent chunk (2048 ký tự).
  - Kết quả: Context đưa vào LLM giữ trọn vẹn toàn bộ bảng biểu, câu trả lời sinh ra chính xác 100% (*"Tổng Giám đốc (CEO) phê duyệt"*), điểm Faithfulness tăng vọt từ 0.70 lên **0.8742**.

### 2. Tắc nghẽn thời gian gọi API khi làm giàu dữ liệu (Enrichment Bottleneck)
- **Hiện tượng:** Có 119 child chunks cần gọi LLM làm giàu. Nếu gọi tuần tự (sequential synchronous calls), thời gian chờ mất tới gần 3 phút, làm chậm toàn bộ quy trình CI/CD và test bài lab.
- **Cách xử lý:** 
  - Áp dụng `concurrent.futures.ThreadPoolExecutor(max_workers=8)` trong hàm `enrich_chunks` (`src/m5_enrichment.py`).
  - Gộp 4 tác vụ (Summary, HyQA, Contextual Prepend, Metadata) vào hàm `_enrich_single_call` với prompt JSON có cấu trúc.
  - Thời gian xử lý toàn bộ 119 chunks giảm ngoạn mục xuống còn **48.9 giây** (tốc độ tăng gấp ~4 lần), đồng thời tiết kiệm 75% số lượng request so với gọi 4 hàm riêng rẽ.

### 3. Trùng lặp và tải lại mô hình nặng gây trễ (Model Loading Overhead)
- **Hiện tượng:** Trong các bài test unit (`test_m3.py`), mỗi test case khởi tạo một instance `CrossEncoderReranker()`, khiến mô hình 2.3GB `bge-reranker-v2-m3` bị load đi load lại 5 lần, làm test mất hơn 50 giây.
- **Cách xử lý:**
  - Thiết kế cache ở cấp độ module `_MODEL_CACHE = {}` trong `src/m3_rerank.py`. Các lần khởi tạo sau đều tái sử dụng instance mô hình đã load trong RAM/VRAM.
  - Sau khi tối ưu, thời gian chạy reranking chỉ còn dưới 1 giây cho toàn bộ bài kiểm tra.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý Pháp lý & Tra cứu Quy định Doanh nghiệp (Enterprise Legal & Policy RAG)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng phương pháp Naive RAG cơ bản với LangChain: đọc file PDF/DOCX thô, cắt đoạn cố định 1000 ký tự (fixed-size chunking), embed bằng mô hình cơ bản và tìm kiếm cosine similarity thuần túy trong FAISS.
- **Vấn đề / Bottlenecks đang gặp:**
  1. *Retrieval Failure:* Khi người dùng hỏi bằng từ ngữ thông dụng (ví dụ: "chế độ ốm đau"), vector search không match tốt với các điều khoản dùng thuật ngữ pháp lý ("chế độ bảo hiểm xã hội khi nghỉ việc do bệnh lý").
  2. *Bị cắt cụt điều khoản:* Các điều luật hoặc bảng biểu quyền lợi bị cắt ngang qua ranh giới chunk, khiến câu trả lời bị thiếu điều kiện loại trừ hoặc thiếu thẩm quyền xử lý.
  3. *Xung đột phiên bản:* Doanh nghiệp cập nhật quy chế qua các năm (2023, 2024), hệ thống thường xuyên trích dẫn nhầm quy chế cũ đã hết hiệu lực.

#### 2. Kế hoạch cải tiến công nghệ
1. **Chunking Strategy:**
   - Chuyển sang **Hierarchical Chunking (Parent 2048 / Child 300)** kết hợp **Structure-aware**: Cắt tài liệu theo từng Điều/Khoản trong văn bản pháp quy. Dùng Child chunk để index và đưa Parent chunk chứa toàn bộ Điều luật vào context để đảm bảo tính pháp lý nguyên vẹn.
2. **Search Retrieval:**
   - Áp dụng **Hybrid Search (BM25 + Dense Qdrant)** kết hợp tách từ tiếng Việt chuẩn xác bằng `underthesea`.
   - Kết hợp điểm số bằng **Reciprocal Rank Fusion (RRF)**: Đảm bảo vừa tìm trúng chính xác các mã số văn bản/số tiền (thế mạnh của BM25), vừa hiểu được ngữ nghĩa mở rộng của câu hỏi (thế mạnh của Dense).
3. **Reranking:**
   - Tích hợp mô hình Cross-encoder `BAAI/bge-reranker-v2-m3` sau tầng retrieval để sàng lọc từ top-25 xuống top-3 điều khoản liên quan nhất trước khi gửi cho LLM.
4. **Metadata & Temporal Filtering:**
   - Thêm metadata tự động ở khâu ingestion: `document_type`, `effective_date`, `status` (active / superseded). Thiết lập bộ lọc mặc định chỉ tìm kiếm các văn bản đang có hiệu lực.
5. **Evaluation & Monitoring:**
   - Xây dựng bộ test benchmark 100 câu hỏi nghiệp vụ và tích hợp **RAGAS** vào pipeline CI/CD để giám sát 4 chỉ số (Faithfulness, Relevancy, Precision, Recall) trước mỗi lần deploy phiên bản mới.

#### 3. Timeline triển khai (4 tuần)
- **Tuần 1:** Xây dựng lại module Document Ingestion: tích hợp Structure-aware + Hierarchical Chunking và trích xuất Metadata cho toàn bộ kho tài liệu quy định nội bộ.
- **Tuần 2:** Cài đặt Qdrant Vector Database, xây dựng Hybrid Search (BM25 + BGE-M3 + RRF) và kiểm thử độ phủ tìm kiếm trên tập 50 câu hỏi thử nghiệm.
- **Tuần 3:** Tích hợp Cross-Encoder Reranker, hoàn thiện prompt generation với cơ chế Parent Context Expansion và trích dẫn số hiệu văn bản pháp lý.
- **Tuần 4:** Chạy RAGAS benchmark toàn diện, phân tích Bottom-10 failures, tối ưu hóa latency và đóng gói production container bằng Docker.
