# Golden datasets

Sprint 0 cần: 30–50 paper hợp pháp và 30 câu hỏi eval có nhãn. Mỗi file có version trong tên (`retrieval_qa.v1.jsonl`); khi đổi nhãn thì tạo version mới, không sửa đè.

## `retrieval_qa.v1.jsonl`, mỗi dòng một câu hỏi

```json
{"id": "q001", "question": "Which loss does the paper use to handle class imbalance?",
 "paper_dois": ["10.1109/cvpr.2019.00949"],
 "relevant_quotes": ["re-weights the loss by effective number of samples"],
 "answerable": true}
```

- `relevant_quotes`: đoạn trích chuẩn. Script eval map quote → chunk_id sau khi ingest, vì chunk_id thay đổi khi đổi chunker.
- `answerable: false`: câu hỏi không có đáp án trong corpus. Hệ thống phải trả lời "chưa đủ dữ liệu".

## `paper_cards.v1.jsonl`, PaperCard gán tay để đo field accuracy

## `injection.v1.jsonl`, đoạn văn chứa prompt injection để đo tỉ lệ bị chặn
