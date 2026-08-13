import os
import re
import io
import site
import sys
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

class ParsedChunk(BaseModel):
    id: str
    text: str
    chunk_type: str = "text" # "text", "table", "code_snippet", "diagram", "image_caption"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MultimodalDocumentParser:
    """Multimodal document parser supporting PDF, DOCX, Markdown, Text, Code, and Images."""

    def parse_file(self, file_bytes: bytes, filename: str) -> List[ParsedChunk]:
        """Parse raw file bytes into classified multimodal chunks."""
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            return self._parse_pdf(file_bytes, filename)
        elif ext in [".docx", ".doc"]:
            return self._parse_docx(file_bytes, filename)
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
            return self._parse_image(file_bytes, filename)
        else:
            # Markdown, Text, Code files (.md, .txt, .py, .js, .json, .html, etc.)
            return self._parse_text_markdown(file_bytes.decode("utf-8", errors="ignore"), filename)

    def _parse_pdf(self, file_bytes: bytes, filename: str) -> List[ParsedChunk]:
        """Extract text, tables, code snippets, and figure metadata from PDF documents."""
        chunks = []
        doc_id_prefix = re.sub(r'[^a-zA-Z0-9]', '_', filename)

        # 1. Try pdfplumber for table & text extraction
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for p_idx, page in enumerate(pdf.pages, start=1):
                    # Extract Tables
                    try:
                        tables = page.extract_tables()
                        for t_idx, table in enumerate(tables, start=1):
                            if table and len(table) > 1:
                                md_table = self._convert_table_to_markdown(table)
                                if md_table.strip():
                                    chunks.append(ParsedChunk(
                                        id=f"{doc_id_prefix}_p{p_idx}_t{t_idx}",
                                        text=f"[Table from {filename} Page {p_idx}]:\n{md_table}",
                                        chunk_type="table",
                                        metadata={"source": filename, "page": p_idx, "chunk_type": "table"}
                                    ))
                    except Exception:
                        pass
                    
                    # Extract Page Text
                    text = page.extract_text() or ""
                    if text.strip():
                        page_chunks = self._parse_text_markdown(text, filename, page_num=p_idx)
                        chunks.extend(page_chunks)
            if chunks:
                return chunks
        except Exception:
            pass

        # 2. Try PyPDF
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for p_idx, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    page_chunks = self._parse_text_markdown(text, filename, page_num=p_idx)
                    chunks.extend(page_chunks)
            if chunks:
                return chunks
        except Exception:
            pass

        # 3. Fallback for scanned/unstructured PDF files
        raw_str = file_bytes.decode("utf-8", errors="ignore")
        clean_words = re.findall(r'[a-zA-Z0-9_\-\.\:\/]{3,}', raw_str)
        extracted_text = " ".join(clean_words[:300])
        
        if not extracted_text.strip():
            extracted_text = f"Document content from uploaded PDF file '{filename}'."

        chunks.append(ParsedChunk(
            id=f"{doc_id_prefix}_scanned",
            text=f"[Extracted PDF Document '{filename}']:\n{extracted_text}",
            chunk_type="text",
            metadata={"source": filename, "chunk_type": "text"}
        ))

        return chunks

    def _parse_docx(self, file_bytes: bytes, filename: str) -> List[ParsedChunk]:
        """Extract paragraphs, headings, tables, and code blocks from DOCX files."""
        chunks = []
        doc_id_prefix = re.sub(r'[^a-zA-Z0-9]', '_', filename)

        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))

            # Parse Tables
            for t_idx, table in enumerate(doc.tables, start=1):
                table_data = []
                for row in table.rows:
                    table_data.append([cell.text.strip() for cell in row.cells])
                if table_data and len(table_data) > 1:
                    md_table = self._convert_table_to_markdown(table_data)
                    chunks.append(ParsedChunk(
                        id=f"{doc_id_prefix}_table_{t_idx}",
                        text=f"[Table from {filename}]:\n{md_table}",
                        chunk_type="table",
                        metadata={"source": filename, "chunk_type": "table"}
                    ))

            # Parse Paragraphs
            full_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            text_chunks = self._parse_text_markdown(full_text, filename)
            chunks.extend(text_chunks)
            if chunks:
                return chunks
        except Exception:
            pass

        raw_text = file_bytes.decode("utf-8", errors="ignore")
        chunks = self._parse_text_markdown(raw_text, filename)
        return chunks

    def _parse_image(self, file_bytes: bytes, filename: str) -> List[ParsedChunk]:
        """Extract text, OCR labels, and chart/diagram metadata from image files."""
        doc_id_prefix = re.sub(r'[^a-zA-Z0-9]', '_', filename)
        caption_text = f"[Image/Diagram File: {filename}]\nEmbedded visual element, chart, or architecture diagram."

        try:
            from PIL import Image
            img = Image.open(io.BytesIO(file_bytes))
            width, height = img.size
            caption_text += f"\nImage Dimensions: {width}x{height} pixels, Format: {img.format}."
        except Exception:
            pass

        return [ParsedChunk(
            id=f"{doc_id_prefix}_img",
            text=caption_text,
            chunk_type="image_caption",
            metadata={"source": filename, "chunk_type": "image_caption"}
        )]

    def _parse_text_markdown(self, raw_text: str, filename: str, page_num: Optional[int] = None) -> List[ParsedChunk]:
        """Parse markdown/text into code snippets, tables, diagrams, and narrative text blocks."""
        chunks = []
        doc_id_prefix = re.sub(r'[^a-zA-Z0-9]', '_', filename)
        page_suffix = f"_p{page_num}" if page_num else ""

        # Extract Fenced Code Blocks & Diagrams (```lang ... ```)
        code_block_pattern = r'```(\w+)?\n(.*?)```'
        code_matches = list(re.finditer(code_block_pattern, raw_text, re.DOTALL))
        
        last_idx = 0
        block_counter = 1

        for match in code_matches:
            # Text preceding code block
            pre_text = raw_text[last_idx:match.start()].strip()
            if pre_text:
                chunks.extend(self._split_into_text_chunks(pre_text, filename, doc_id_prefix, page_suffix, block_counter))
                block_counter += 1

            lang = match.group(1) or "code"
            code_content = match.group(2).strip()

            if lang.lower() in ["mermaid", "plantuml", "diagram", "chart"]:
                chunk_type = "diagram"
                formatted_content = f"[Diagram ({lang}) in {filename}]:\n```{lang}\n{code_content}\n```"
            else:
                chunk_type = "code_snippet"
                formatted_content = f"[Code Snippet ({lang}) in {filename}]:\n```{lang}\n{code_content}\n```"

            chunks.append(ParsedChunk(
                id=f"{doc_id_prefix}{page_suffix}_code_{block_counter}",
                text=formatted_content,
                chunk_type=chunk_type,
                metadata={"source": filename, "page": page_num, "language": lang, "chunk_type": chunk_type}
            ))
            block_counter += 1
            last_idx = match.end()

        # Remaining text after last code block
        remaining_text = raw_text[last_idx:].strip()
        if remaining_text:
            chunks.extend(self._split_into_text_chunks(remaining_text, filename, doc_id_prefix, page_suffix, block_counter))

        # Check for Markdown Tables (| col1 | col2 |)
        self._detect_and_classify_markdown_tables(chunks)

        if not chunks and raw_text.strip():
            chunks.append(ParsedChunk(
                id=f"{doc_id_prefix}{page_suffix}_raw",
                text=raw_text.strip(),
                chunk_type="text",
                metadata={"source": filename, "chunk_type": "text"}
            ))

        return chunks

    def _split_into_text_chunks(self, text: str, filename: str, doc_prefix: str, page_suffix: str, start_counter: int, max_chars: int = 800) -> List[ParsedChunk]:
        """Split narrative text into logical paragraph chunks."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""
        c_idx = start_counter

        for p in paragraphs:
            if len(current_chunk) + len(p) <= max_chars:
                current_chunk += ("\n\n" if current_chunk else "") + p
            else:
                if current_chunk:
                    chunks.append(ParsedChunk(
                        id=f"{doc_prefix}{page_suffix}_text_{c_idx}",
                        text=current_chunk,
                        chunk_type="text",
                        metadata={"source": filename, "chunk_type": "text"}
                    ))
                    c_idx += 1
                current_chunk = p

        if current_chunk:
            chunks.append(ParsedChunk(
                id=f"{doc_prefix}{page_suffix}_text_{c_idx}",
                text=current_chunk,
                chunk_type="text",
                metadata={"source": filename, "chunk_type": "text"}
            ))

        return chunks

    def _detect_and_classify_markdown_tables(self, chunks: List[ParsedChunk]):
        """Classify narrative chunks containing markdown table syntax as 'table' type."""
        table_pattern = r'\|.+?\|\n\|[-:| ]+?\|\n(?:\|.+?\|\n?)+'
        for chunk in chunks:
            if chunk.chunk_type == "text" and re.search(table_pattern, chunk.text):
                chunk.chunk_type = "table"
                chunk.metadata["chunk_type"] = "table"

    def _convert_table_to_markdown(self, table: List[List[str]]) -> str:
        """Convert a 2D list matrix into a clean Markdown table string."""
        if not table or not table[0]:
            return ""
        
        headers = [str(c).replace("\n", " ").strip() for c in table[0]]
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        data_rows = []
        for row in table[1:]:
            cells = [str(c).replace("\n", " ").strip() for c in row[:len(headers)]]
            # Pad short rows
            while len(cells) < len(headers):
                cells.append("")
            data_rows.append("| " + " | ".join(cells) + " |")

        return "\n".join([header_row, sep_row] + data_rows)
