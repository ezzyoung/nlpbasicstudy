# RAG (Retrieval-Augmented Generation) 뼈대

import torch
import torch.nn as nn


class DocumentEncoder(nn.Module):
    """문서를 벡터로 변환하는 인코더"""

    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return hidden.squeeze(0)  # (batch, hid_dim)


class QueryEncoder(nn.Module):
    """쿼리를 벡터로 변환하는 인코더"""

    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return hidden.squeeze(0)


class Retriever(nn.Module):
    """쿼리와 문서 간 유사도 기반 검색기"""

    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.query_encoder = QueryEncoder(vocab_size, emb_dim, hid_dim)
        self.doc_encoder = DocumentEncoder(vocab_size, emb_dim, hid_dim)

    def forward(self, query, documents):
        # query: (batch, seq_len)
        # documents: (num_docs, seq_len)
        q_vec = self.query_encoder(query)       # (batch, hid_dim)
        d_vecs = self.doc_encoder(documents)    # (num_docs, hid_dim)

        # 내적 유사도
        scores = torch.matmul(q_vec, d_vecs.T)  # (batch, num_docs)
        return scores

    def retrieve(self, query, documents, top_k=3):
        scores = self.forward(query, documents)
        top_k_scores, top_k_indices = torch.topk(scores, k=top_k, dim=-1)
        return top_k_indices, top_k_scores


class Generator(nn.Module):
    """검색된 문서 + 쿼리를 받아 답변을 생성하는 디코더"""

    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input_token, hidden, cell, context):
        # input_token: (batch, 1)
        # context: (batch, hid_dim) - 검색된 문서 정보
        embedded = self.embedding(input_token)  # (batch, 1, emb_dim)
        context = context.unsqueeze(1)          # (batch, 1, hid_dim)

        lstm_input = torch.cat([embedded, context], dim=-1)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))  # (batch, vocab_size)
        return prediction, hidden, cell


class RAG(nn.Module):
    """Retrieval-Augmented Generation 전체 파이프라인"""

    def __init__(self, vocab_size, emb_dim, hid_dim, top_k=3):
        super().__init__()
        self.retriever = Retriever(vocab_size, emb_dim, hid_dim)
        self.generator = Generator(vocab_size, emb_dim, hid_dim)
        self.top_k = top_k
        self.hid_dim = hid_dim

    def forward(self, query, documents, target=None, max_len=50, sos_token=1, eos_token=2):
        batch_size = query.shape[0]

        # 1. Retrieve: 관련 문서 검색
        top_k_indices, _ = self.retriever.retrieve(query, documents, self.top_k)

        # 2. 검색된 문서 벡터를 평균내서 context 생성
        doc_vecs = self.retriever.doc_encoder(documents)  # (num_docs, hid_dim)
        retrieved_vecs = doc_vecs[top_k_indices]           # (batch, top_k, hid_dim)
        context = retrieved_vecs.mean(dim=1)               # (batch, hid_dim)

        # 3. Generate: context를 조건으로 답변 생성
        hidden = torch.zeros(1, batch_size, self.hid_dim, device=query.device)
        cell = torch.zeros(1, batch_size, self.hid_dim, device=query.device)
        outputs = []

        if target is not None:
            # 학습: teacher forcing
            for t in range(target.shape[1]):
                input_token = target[:, t].unsqueeze(1)
                pred, hidden, cell = self.generator(input_token, hidden, cell, context)
                outputs.append(pred)
        else:
            # 추론: autoregressive
            input_token = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=query.device)
            for _ in range(max_len):
                pred, hidden, cell = self.generator(input_token, hidden, cell, context)
                outputs.append(pred)
                input_token = pred.argmax(dim=-1).unsqueeze(1)
                if (input_token.squeeze(1) == eos_token).all():
                    break

        return torch.stack(outputs, dim=1)  # (batch, seq_len, vocab_size)
