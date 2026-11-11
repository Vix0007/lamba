#!/usr/bin/env python3
"""BM quality gate for training TARGETS (chosen / SFT response / replay sequence), from the native-editor audit
(workflow wf_3457deed-3f4: 74/252 sampled Malay rows had errors: wrong language/script, other-model self-ID,
Indonesian lexicon ~8%+, calques, invented tatabahasa). A sovereign Malaysian model must not learn broken BM.
Usage (library): from bm_gate import gate; reasons = gate(prompt, answer_text, lang)  -> [] if clean.
Batch fastText scoring is done once via prescore(list_of_texts)."""
import re, json, subprocess
D = "/home/ubuntu/antidoom_data/"
BAD_SCRIPT = re.compile(r"[ऀ-ॿ฀-๿぀-ヿ一-鿿가-힯؀-ۿ]")
OTHER_SELF = re.compile(r"\b(aku|saya|i am|i'?m|my name is|nama (saya|aku))\s+(ialah\s+|adalah\s+|is\s+)?\**(qwen|tongyi|chatgpt|gpt-?\d?|claude|gemini|deepseek|llama|grok|ernie|kimi|mistral|copilot)\b|"
                        r"\b(dibangunkan|dibina|dicipta|developed|built|created|made) (oleh|by) \**(alibaba|openai|google|anthropic|meta|bytedance|deepseek|tongyi)\b(?![^.\n]{0,40}\b(ialah|is a|adalah|merupakan)\b)", re.I)
WRONG_MAKER = re.compile(r"bytedance|\bvix(?!ero)[a-z]*\b|lamba vanga|lamba labs|vixero (?!labs)\w", re.I)
# Indonesian lexicon (expanded from the audit), whole words; Malay equivalents in comments
INDO = re.compile(r"\b(nggak|gak|enggak|banget|udah|gimana|aja|kok|lho|sih|dong|nih|kayak|gitu|silakan|asisten|berbasis|otomatis|kedaluwarsa|"
                  r"dikembangkan|mengembangkan|perusahaan|arsitektur|komputasi|fitur|dibangun|jawaban|jawabannya|cek|optimasi|sehari-hari|bikin|"
                  r"penerjemahan|pemrograman|mau|sebetulnya|diseduh|rangkuman|terstruktur|akurat|ekspektasi|koreksi|perhitungan|perhitungannya|"
                  r"beradaptasi|oke|butuh|barusan|insting|terintegrasi|resmi|bisa|kasih tahu|cari tahu|kamu bisa|tidak bisa|anda bisa|saya bisa|aku bisa)\b", re.I)
TATABAHASA = re.compile(r"tatabahasa|kata pemeri|kata hubung|kata sendi|imbuhan|kata nama|kata kerja|kata adjektif|frasa (nama|kerja|adjektif)|"
                        r"penanda wacana|ayat (majmuk|tunggal|pasif|aktif)|kata ganda|peribahasa|simpulan bahasa|kata pinjaman|morfologi|sintaksis", re.I)
HINGLISH = re.compile(r"\b(hai|hain|ek|jo|ne|ki|ka|aur|nahi|nahin|kya|haan|mein|bhi|kar|karta|karti|wala|wali)\b", re.I)
def prescore(texts):
    texts = [t for t in set(texts) if t]
    if not texts: return {}, {}
    run = lambda s: json.loads(subprocess.run(["/home/ubuntu/venv-ft312/bin/python", D + s], input=json.dumps(texts), capture_output=True, text=True, check=True).stdout)
    return dict(zip(texts, run("ms_id_score.py"))), dict(zip(texts, run("ms_other_score.py")))
def is_malay_prompt(p):
    return bool(re.search(r"\b(apa|apakah|ini|itu|tu|ni|saya|awak|anda|kau|aku|boleh|tak|tidak|yang|dengan|untuk|macam mana|bagaimana|kenapa|mengapa|ke)\b", p, re.I)) and not re.search(r"\b(the|what|how|why|is|are|you|your)\b", p, re.I)
def gate(prompt, answer, lang, SC=None, OC=None, is_tool=False):
    """Return a list of failure reasons for a TRAINING TARGET answer (visible answer, not the thinking)."""
    r = []
    a = re.sub(r"<tool_call>.*", "", answer or "", flags=re.S).strip()
    if BAD_SCRIPT.search(a) and not BAD_SCRIPT.search(prompt or ""): r.append("bad_script")
    if OTHER_SELF.search(a): r.append("other_model_self_id")
    if WRONG_MAKER.search(a): r.append("wrong_maker")
    ms = lang == "ms" or (lang is None and is_malay_prompt(prompt or ""))
    if ms and a:
        if INDO.search(a): r.append("indo_lexicon")
        if len(HINGLISH.findall(a)) >= 3: r.append("hinglish")
        if SC is not None and SC.get(a, 0) >= 0.5: r.append("indo_fasttext")
        if OC is not None and not is_tool and OC.get(a, 0) >= 0.6 and len(a.split()) >= 6: r.append("not_malay")
    return r
