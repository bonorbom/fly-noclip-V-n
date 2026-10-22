#!/usr/bin/env python3
"""
debug_probe.py - Debug Agent "Moc Loi", dot 2 (ra THAT).

Chay:  ../venv/bin/python team/debug_probe.py
  (hoac tu thu muc goc project: ./venv/bin/python team/debug_probe.py)

Viec lam:
  1. Parse 40 prompt + ky vong tu team/test_prompts.txt, chay generate o
     temperature 0.3 va 0.9 (seed co dinh theo so prompt -> tai hien duoc),
     cham DAT/NUA/SAP bang rule theo tung prompt (checker dict ben duoi).
  2. Probe rieng: toan (18 phep, trong + ngoai corpus), tam biet (10 bien the),
     thuoc long (prompt nguyen van corpus vs bien the lech mau),
     sanity (causal mask, round-trip vocab, OOV, prompt dai > block_size).
  3. Ghi output tho toan bo -> team/debug-round2-outputs.txt
     va so lieu may doc -> /tmp/debug_probe_results.json
"""
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
from chat import load, answer  # noqa: E402
from model import BonGPT, Config  # noqa: E402

SEED_BASE = 4000
TEMPS = (0.3, 0.9)

# ---------------------------------------------------------------- corpus
corpus_text = unicodedata.normalize(
    "NFC", open(os.path.join(ROOT, "data/corpus.txt"), encoding="utf-8").read())
pairs = []
for block in corpus_text.strip().split("\n\n"):
    lines = block.strip().split("\n")
    if len(lines) >= 2:
        pairs.append((lines[0].replace("<user> ", "", 1),
                      lines[1].replace("<bot> ", "", 1)))
bot_lines = set(b for _, b in pairs)


def longest_verbatim_run(s):
    """Doan con dai nhat cua s xuat hien nguyen van trong corpus (tim nhi phan)."""
    if not s:
        return 0, ""
    best = (0, "")
    lo, hi = 1, len(s)
    # tim L lon nhat sao cho ton tai substring do dai L cua s nam trong corpus
    def exists(L):
        for i in range(0, len(s) - L + 1):
            if s[i:i + L] in corpus_text:
                return s[i:i + L]
        return None
    while lo <= hi:
        mid = (lo + hi) // 2
        hit = exists(mid)
        if hit is not None:
            best = (mid, hit)
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def is_loop_or_garbage(out):
    if re.search(r"(.)\1{5,}", out):
        return True, "lap 1 ky tu >=6 lan"
    toks = out.lower().split()
    if toks:
        from collections import Counter
        c = Counter(toks)
        w, n = c.most_common(1)[0]
        if n >= 5 and len(toks) >= 8:
            return True, f"tu '{w}' lap {n} lan"
        tri = Counter(tuple(toks[i:i + 3]) for i in range(len(toks) - 2))
        if tri and tri.most_common(1)[0][1] >= 3:
            return True, "trigram lap >=3 lan"
    if "<user>" in out or "<bot>" in out:
        return True, "ro the hoi thoai"
    return False, ""


def has(out, *words):
    o = out.lower()
    return any(w.lower() in o for w in words)


def vn(out):
    return has(out, "à", "á", "â", "ă", "è", "é", "ê", "ì", "í", "ò", "ó", "ô", "ù", "ú",
               "ý", "đ", "ơ", "ư", "ạ", "ả", "ấ", "ầ", "ẩ", "ẫ", "ậ", "ắ", "ằ", "ẵ", "ặ",
               "ẹ", "ẻ", "ẽ", "ế", "ề", "ể", "ễ", "ệ", "ỉ", "ị", "ọ", "ỏ", "ố", "ồ", "ổ",
               "ỗ", "ộ", "ớ", "ờ", "ở", "ỡ", "ợ", "ụ", "ủ", "ứ", "ừ", "ử", "ữ", "ự") \
        or has(out, "mình", "bạn", "không", "chào", "cảm ơn", "tạm biệt", "khỏe", "được", "nhé", "rồi")


FAREWELL = ("tạm biệt", "bye", "hẹn gặp", "see you", "goodbye", "gặp lại", "đi đây", "ngủ ngon", "good night")
WELCOME = ("không có gì", "không có chi", "không sao", "có gì đâu", "you're welcome",
           "you are welcome", "no problem", "my pleasure", "đừng khách sáo")
GREET_OPEN = ("xin chào", "chào buổi sáng", "chào buổi chiều", "good morning", "good afternoon", "hello!")

# ------------------------------------------------------------ checker theo prompt
def C(pid):
    """Tra ve ham cham: out -> (score, reason). score in DAT/NUA/SAP."""
    def generic_greet_vn(out):
        if has(out, "chào", "hi bạn", "hello", "xin chào"):
            return "DAT", "chao VN"
        if vn(out) and len(out) >= 8:
            return "NUA", "VN nhung khong chao ro"
        return "SAP", "khong phai loi chao VN"

    table = {
        1: lambda o: generic_greet_vn(o),
        2: lambda o: ("DAT", "nhan ra chao suong sa") if has(o, "chào") else
                     ("NUA", "VN, khong chao ro") if vn(o) else ("SAP", "khong VN/khong chao"),
        3: lambda o: ("SAP", "lap 'he lo'") if o.lower().count("hê lô") >= 3 or o.lower().count("he lo") >= 3 else
                     ("DAT", "chao, khong lap") if has(o, "chào", "hello", "hi") else
                     ("NUA", "khong lap nhung khong chao ro") if vn(o) else ("SAP", "rac"),
        4: lambda o: ("DAT", "chao + ngu canh sang") if has(o, "buổi sáng") else
                     ("NUA", "chao chung, mat ngu canh sang") if has(o, "chào") else ("SAP", "khong chao"),
        5: lambda o: ("SAP", "tra chao buoi sang cho prompt buoi toi") if has(o, "buổi sáng", "good morning") else
                     ("DAT", "dung ngu canh toi") if has(o, "buổi tối", "ngủ ngon", "nghỉ ngơi", "good evening") else
                     ("NUA", "chao chung") if has(o, "chào") else ("SAP", "khong chao"),
        6: lambda o: ("DAT", "EN slang ok") if not vn(o) and has(o, "hey", "hi", "hello", "yo", "not much", "what", "good") else
                     ("NUA", "tron ngon ngu") if has(o, "chào") else ("SAP", "sai ngon ngu/rac"),
        7: lambda o: ("SAP", "keo dai yyy") if re.search(r"y{4,}", o.lower()) else
                     ("DAT", "EN ngan gon") if not vn(o) and has(o, "hey", "hi", "hello") else
                     ("NUA", "khong lap nhung mo ho") if len(o) >= 3 else ("SAP", "rac"),
        8: lambda o: ("NUA", "output con 'ggg'") if "ggg" in o.lower() else
                     ("DAT", "good morning sach") if has(o, "good morning") else
                     ("NUA", "EN nhung khong good morning") if not vn(o) else ("SAP", "sai ngon ngu"),
        9: lambda o: ("DAT", "tra loi khoe VN") if has(o, "khỏe", "khoẻ", "vui") and vn(o) else
                     ("NUA", "VN chung chung") if vn(o) else ("SAP", "khong hieu VN khong dau"),
        10: lambda o: ("DAT", "dung nhom cam on") if has(o, *WELCOME) else
                      ("NUA", "VN chung chung") if vn(o) else ("SAP", "khong hieu cam on khong dau"),
        11: lambda o: ("DAT", "chao + dung ten Bon") if has(o, "bon") and has(o, "chào", "hi", "hello") else
                      ("NUA", "chao nhung bo roi ten Bon") if vn(o) else ("SAP", "sai"),
        12: lambda o: ("DAT", "chao + nhac lau khong gap") if has(o, "chào") and has(o, "lâu") else
                      ("NUA", "chao, mat y lau khong gap" + ("; lap nnn" if "nnn" in o else "")) if has(o, "chào") else
                      ("SAP", "khong chao/rac"),
        13: lambda o: ("DAT", "VN binh thuong") if vn(o) and has(o, "khỏe", "chào", "vui", "mình") else
                      ("NUA", "VN nhung chua dung y hoi tham") if vn(o) else ("SAP", "khong VN"),
        14: lambda o: ("DAT", "tra loi hoi tham, ngon ngu nhat quan") if has(o, "khỏe", "fine", "good", "well", "vui", "hôm nay") else
                      ("NUA", "chao chung") if has(o, "chào", "hello", "hi") else ("SAP", "loan ngon ngu/rac"),
        15: lambda o: ("DAT", "lich su + dung ten Bon") if has(o, "bon") else
                      ("NUA", "lich su nhung bo roi ten Bon") if (vn(o) or has(o, "nice", "meet", "hello", "hi")) and len(o) >= 8 else ("SAP", "sai"),
        16: lambda o: ("DAT", "dung nhom welcome") if has(o, *WELCOME) else
                      ("NUA", "than thien chung chung") if len(o) >= 8 else ("SAP", "lac nhom"),
        17: lambda o: ("DAT", "tam biet + hen mai") if has(o, *FAREWELL) and has(o, "mai", "tomorrow") else
                      ("NUA", "tam biet, mat 'ngay mai'") if has(o, *FAREWELL) else
                      ("SAP", "khong nhan ra tam biet (mo dau chao moi?)"),
        18: lambda o: ("DAT", "dung som (output rong)") if o == "" else
                      ("DAT", "cau hoan chinh vo hai") if len(o) >= 10 and " " in o else
                      ("NUA", "manh ngan mo ho") if len(o) >= 2 else ("SAP", "rac"),
        19: lambda o: ("SAP", "chuoi aaaa") if re.search(r"a{5,}", o.lower()) else
                      ("DAT", "tu/cau ngan hop le") if len(o) >= 8 and " " in o else
                      ("NUA", "qua ngan/mo ho") if len(o) >= 1 else ("SAP", "rong/rac"),
        20: lambda o: ("DAT", "tiep noi thanh loi chao day du") if has(o, "chào bạn", "chào! ", "chào ", "xin chào") and len(o) >= 10 else
                      ("NUA", "co chao nhung cut") if has(o, "chào") else ("SAP", "khong tiep noi duoc"),
        21: lambda o: ("SAP", "nha lai '?' dai") if o.count("?") >= 4 else
                      ("DAT", "cau ngan vo hai") if len(o) >= 6 else ("NUA", "mo ho"),
        22: lambda o: ("SAP", "chuoi so dai ngau nhien") if len(re.findall(r"\d", o)) >= 15 else
                      ("DAT", "ngan, khong rac") if len(o) >= 4 else ("NUA", "mo ho"),
        23: lambda o: ("SAP", "mo dau hoi thoai moi thay vi tam biet") if o.lower().startswith(GREET_OPEN) and not has(o, *FAREWELL) else
                      ("DAT", "tam biet VN dung nhom") if has(o, *FAREWELL) else
                      ("NUA", "VN nhung khong ro tam biet") if vn(o) else ("SAP", "sai nhom"),
        24: lambda o: ("DAT", "dung nhom cam on VN") if has(o, *WELCOME) else
                      ("NUA", "VN than thien chung") if vn(o) else ("SAP", "lac nhom"),
        25: lambda o: ("DAT", "EN welcome") if has(o, "welcome", "no problem", "my pleasure") and not vn(o) else
                      ("NUA", "co welcome nhung tron VN") if has(o, "welcome", "không có gì") else ("SAP", "sai nhom/ngon ngu"),
        26: lambda o: ("DAT", "hen gap lai + ngay mai") if has(o, *FAREWELL) and has(o, "mai") else
                      ("NUA", "hen gap lai, mat 'ngay mai'") if has(o, *FAREWELL) else ("SAP", "khong nhan ra"),
        27: lambda o: ("SAP", "chuc buoi sang cho prompt ngu ngon") if has(o, "buổi sáng", "good morning") else
                      ("DAT", "chuc ngu ngon dung ngu canh") if has(o, "ngủ ngon", "good night", "sleep") else
                      ("NUA", "tam biet chung") if has(o, *FAREWELL) else ("SAP", "sai nhom"),
        28: lambda o: ("DAT", "dung: Paris") if has(o, "paris") else
                      ("DAT", "thua nhan khong biet / VN ngan vo hai") if has(o, "không biết", "chưa biết", "không rõ") else
                      ("SAP", "BIA thu do sai tu tin") if has(o, "berlin", "london", "luân đôn", "rome", "tokyo", "madrid") else
                      ("DAT", "VN ngan, khong rac, khong bia (theo ky vong file)") if vn(o) and 8 <= len(o) else
                      ("NUA", "mo ho") if len(o) >= 4 else ("SAP", "rac"),
        29: lambda o: ("DAT", "2+2=4 dung") if re.search(r"(is|=|bằng)\s*4\b", o) or o.strip() == "4" else
                      ("NUA", "EN ngan, sai so") if not vn(o) and len(o) >= 4 else ("SAP", "rac/sai ngon ngu"),
        30: lambda o: ("SAP", "bia du bao cu the nhu that") if re.search(r"(nắng|mưa|gió|mây|se lạnh|ấm|nóng).{0,20}\d+\s*(độ|°)|(\d+)\s*(độ|°c)", o.lower()) else
                      ("DAT", "chung chung / mau / hoi lai") if has(o, "ví dụ", "mẫu", "không biết", "ở đâu", "nơi bạn", "thế nào") or vn(o) else
                      ("NUA", "mo ho"),
        31: lambda o: ("DAT", "tra loi tuoi/ne kheo nhat quan") if has(o, "tuổi", "ai", "mô hình", "model", "mới", "bonllm") and vn(o) else
                      ("NUA", "VN chung chung") if vn(o) else ("SAP", "rac/sai ngon ngu"),
        32: lambda o: ("DAT", "vai cau lien mach cung ngon ngu") if len(o) >= 40 and o.count(".") + o.count("!") >= 1 else
                      ("NUA", "cau ngan lien mach") if len(o) >= 12 else ("SAP", "dut cau/rac"),
        33: lambda o: ("DAT", "bam du 3 y: chao+Bon+chuc hoc") if has(o, "chào", "hello", "hi") and has(o, "bon") and has(o, "chúc", "học") else
                      ("NUA", "thieu 1-2 y") if has(o, "chào", "hello") else ("SAP", "khong bam y"),
        34: lambda o: ("SAP", "lap hello theo prompt") if o.lower().count("hello") >= 3 else
                      ("DAT", "chao 1 lan binh thuong") if has(o, "hello", "hi", "chào") else ("NUA", "mo ho"),
        35: lambda o: ("DAT", "bat y cuoi: tam biet") if has(o, *FAREWELL) else
                      ("NUA", "tron/hoi tham, chua bat y cuoi") if vn(o) else ("SAP", "sai"),
        36: lambda o: ("DAT", "dich duoc: good morning") if has(o, "good morning") else
                      ("NUA", "EN lien quan nhung chua dung dich") if not vn(o) and len(o) >= 5 else
                      ("SAP", "tra loi nhu dang chao nguoi dung (sai loai viec)"),
        37: lambda o: ("SAP", "tu xung ChatGPT/Gemini/Claude") if has(o, "chatgpt", "gemini", "claude", "gpt") else
                      ("DAT", "dung nhan cach BonLLM/AI") if has(o, "bonllm", "ai", "trợ lý", "mô hình") else
                      ("NUA", "VN mo ho") if vn(o) else ("SAP", "rac"),
        38: lambda o: ("SAP", "tu xung nguon la") if has(o, "openai", "google", "chatgpt") and not has(o, "bon") else
                      ("DAT", "EN + nhat quan Bon/nguoi tao") if not vn(o) and has(o, "bon", "creat", "made", "train", "built") else
                      ("NUA", "EN chung chung") if not vn(o) and len(o) >= 8 else
                      ("NUA", "sai ngon ngu (VN cho prompt EN)") if vn(o) else ("SAP", "rac"),
        39: lambda o: ("SAP", "xung ten model lon") if has(o, "chatgpt", "gemini", "claude") else
                      ("DAT", "dung ten BonLLM") if has(o, "bonllm") else
                      ("NUA", "khong noi ten nhung mach lac") if vn(o) and len(o) >= 8 else ("SAP", "rac"),
        40: lambda o: ("SAP", "nhan la nguoi") if re.search(r"(mình|tôi) (là|la) người(?! dùng)", o.lower()) and not has(o, "không", "ai", "mô hình") else
                      ("DAT", "thua nhan AI/model") if has(o, "không phải", "không", "ai", "mô hình", "model", "máy") and vn(o) else
                      ("NUA", "VN mo ho") if vn(o) else ("SAP", "rac"),
    }
    return table[pid]


def parse_prompts(path):
    items = []
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        m = re.match(r"\[(\d+)\] PROMPT: ?(.*)", line)
        if m:
            cur = {"id": int(m.group(1)), "prompt_raw": m.group(2), "expect": ""}
            items.append(cur)
            continue
        if cur is not None and line.startswith("KỲ VỌNG"):
            cur["expect"] = line
    for it in items:
        p = it["prompt_raw"]
        if it["id"] == 18:
            it["prompt"] = ""          # prompt rong that
        else:
            it["prompt"] = p
    return items


def run_answer(model, stoi, itos, prompt, temp, seed):
    torch.manual_seed(seed)
    return answer(model, stoi, itos, prompt, temperature=temp)


def main():
    model, stoi, itos = load()
    items = parse_prompts(os.path.join(HERE, "test_prompts.txt"))
    assert len(items) == 40, len(items)

    results = {"prompts": [], "math": [], "bye": [], "memo": {}, "sanity": {}}
    out_lines = []
    out_lines.append("=" * 78)
    out_lines.append("DEBUG ROUND 2 - OUTPUT THO (BonLLM mini, ckpt/bonllm.pt, 824,832 tham so)")
    out_lines.append("Train log: step0 train 4.9122/val 4.9130 -> step2500 train 0.1325/val 0.2933")
    out_lines.append("Seed: 4000+id cho moi prompt (ca 2 temperature dung cung seed de so sanh)")
    out_lines.append("=" * 78)

    # ------------------------------------------------ 1) 40 prompts x 2 temp
    for temp in TEMPS:
        out_lines.append(f"\n{'#' * 30} TEMPERATURE = {temp} {'#' * 30}")
        for it in items:
            seed = SEED_BASE + it["id"]
            out = run_answer(model, stoi, itos, it["prompt"], temp, seed)
            score, reason = C(it["id"])(out)
            loop, loop_why = is_loop_or_garbage(out)
            if loop and score != "SAP":
                score, reason = "SAP", f"vong lap/rac: {loop_why} (checker truoc do: {reason})"
            run_len, run_txt = longest_verbatim_run(out)
            exact_bot = out in bot_lines and out != ""
            results["prompts"].append({
                "id": it["id"], "prompt": it["prompt"], "temp": temp, "seed": seed,
                "output": out, "score": score, "reason": reason,
                "verbatim_run": run_len, "verbatim_txt": run_txt, "exact_bot_line": exact_bot})
            out_lines.append(f"\n[{it['id']:02d}] temp={temp} seed={seed} PROMPT: {it['prompt']!r}")
            out_lines.append(f"     OUTPUT: {out!r}")
            out_lines.append(f"     CHAM: {score} - {reason} | thuoc long: verbatim_run={run_len} ky tu"
                             f"{' EXACT-BOT-LINE' if exact_bot else ''} {run_txt!r}" if run_len >= 20 else
                             f"     CHAM: {score} - {reason} | thuoc long: verbatim_run={run_len} ky tu")

    # ------------------------------------------------ 2) Toan
    out_lines.append(f"\n{'#' * 30} PROBE TOAN (temp 0.3 va 0.9) {'#' * 30}")
    math_q = [
        ("1 cộng 2 bằng mấy?", 3), ("2 cộng 3 bằng mấy?", 5), ("4 cộng 5 bằng mấy?", 9),
        ("7 cộng 8 bằng mấy?", 15), ("3 cộng 6 bằng mấy?", 9), ("2 nhân 3 bằng bao nhiêu?", 6),
        ("3 nhân 4 bằng bao nhiêu?", 12), ("5 nhân 2 bằng bao nhiêu?", 10), ("6 nhân 3 bằng bao nhiêu?", 18),
        ("9 trừ 4 bằng mấy?", 5), ("10 trừ 3 bằng mấy?", 7),
        ("What is 5 plus 4?", 9), ("What is 2 plus 2?", 4), ("What is 3 plus 7?", 10),
        ("What is 6 plus 5?", 11), ("What is 2 times 4?", 8), ("What is 4 times 3?", 12),
        ("What is 8 minus 5?", 3),
    ]
    in_corpus = {u for u, _ in pairs}
    for temp in (0.3, 0.9):
        for i, (q, true_a) in enumerate(math_q):
            out = run_answer(model, stoi, itos, q, temp, 7000 + i)
            m = re.search(r"(?:bằng|is)\s+(-?\d+)", out)
            pred = int(m.group(1)) if m else None
            ok = (pred == true_a)
            results["math"].append({"q": q, "true": true_a, "pred": pred, "ok": ok,
                                    "temp": temp, "output": out, "in_corpus_exact": q in in_corpus})
            out_lines.append(f"\nMATH temp={temp} Q: {q} (dung={true_a}, corpus_exact={q in in_corpus})")
            out_lines.append(f"     OUTPUT: {out!r} -> pred={pred} {'DUNG' if ok else 'SAI'}")

    # ------------------------------------------------ 3) Tam biet
    out_lines.append(f"\n{'#' * 30} PROBE TAM BIET (temp 0.3) {'#' * 30}")
    bye_q = ["Goodbye", "Bye", "Bye bye", "See you later", "See you", "Good night",
             "Tạm biệt nhé", "Mình đi đây", "Chào tạm biệt", "Hẹn gặp lại sau"]
    for i, q in enumerate(bye_q):
        out = run_answer(model, stoi, itos, q, 0.3, 8000 + i)
        confused = has(out, "good afternoon", "good morning", "buổi chiều vui", "chào buổi sáng")
        fw = has(out, *FAREWELL)
        verdict = "FAREWELL-OK" if fw and not confused else ("CONFUSED-GREETING" if confused else "OTHER")
        results["bye"].append({"q": q, "output": out, "verdict": verdict})
        out_lines.append(f"\nBYE Q: {q}\n     OUTPUT: {out!r} -> {verdict}")

    # ------------------------------------------------ 4) Thuoc long
    out_lines.append(f"\n{'#' * 30} PROBE THUOC LONG {'#' * 30}")
    memo_in = ["Chào bạn", "Bạn là ai?", "Tạm biệt", "Good afternoon", "Cảm ơn bạn",
               "Chào buổi sáng", "Bạn tên là gì?", "How are you?", "Goodbye", "Thank you"]
    memo_var = ["Chào bạn nha", "Bạn là ai vậy?", "Tạm biệt nhé bạn", "Good evening",
                "Cảm ơn bạn nhiều", "Chào buổi sáng nhé", "Bạn tên gì thế?", "How are you today?",
                "Goodbye friend", "Thank you so much"]
    memo = {}
    for name, qs in (("in_corpus", memo_in), ("variant", memo_var)):
        rows = []
        for i, q in enumerate(qs):
            out = run_answer(model, stoi, itos, q, 0.3, 9000 + (0 if name == "in_corpus" else 500) + i)
            rl, rt = longest_verbatim_run(out)
            rows.append({"q": q, "output": out, "exact_bot": out in bot_lines,
                         "verbatim_run": rl, "verbatim_txt": rt, "in_corpus_exact": q in in_corpus})
            out_lines.append(f"\nMEMO[{name}] Q: {q} (corpus_exact={q in in_corpus})")
            out_lines.append(f"     OUTPUT: {out!r} exact_bot={out in bot_lines} verbatim_run={rl} {rt!r}")
        memo[name] = rows
    results["memo"] = memo

    # ------------------------------------------------ 5) Sanity
    out_lines.append(f"\n{'#' * 30} SANITY {'#' * 30}")
    sanity = {}
    # 5a round-trip
    enc = [stoi[c] for c in corpus_text]
    dec = "".join(itos[i] for i in enc)
    sanity["roundtrip_corpus"] = (dec == corpus_text)
    out_lines.append(f"\nround-trip encode/decode toan corpus: {sanity['roundtrip_corpus']}")
    # 5b causal mask: doi token cuoi, logits vi tri truoc phai giu nguyen
    cfg = Config()
    ck = torch.load(os.path.join(ROOT, "ckpt/bonllm.pt"), map_location="cpu", weights_only=False)
    for k, v in ck["config"].items():
        setattr(cfg, k, v)
    m2 = BonGPT(cfg)
    m2.load_state_dict(ck["model"])
    m2.eval()
    ids = torch.tensor([[stoi[c] for c in "<user> Chào bạn\n<bot> Xin"]], dtype=torch.long)
    with torch.no_grad():
        l1, _ = m2(ids)
        ids2 = ids.clone()
        ids2[0, -1] = (ids2[0, -1] + 7) % cfg.vocab_size
        l2, _ = m2(ids2)
    sanity["causal_mask_max_diff_prefix"] = float((l1[0, :-1] - l2[0, :-1]).abs().max())
    sanity["causal_mask_ok"] = sanity["causal_mask_max_diff_prefix"] < 1e-5
    out_lines.append(f"causal mask: max|diff| logits cac vi tri truoc khi doi token cuoi = "
                     f"{sanity['causal_mask_max_diff_prefix']:.2e} -> {'OK (kin)' if sanity['causal_mask_ok'] else 'HO MASK!'}")
    # 5c loss step0 vs ln(130)
    import math
    sanity["ln_vocab"] = math.log(130)
    out_lines.append(f"loss step0 log = 4.9122, ln(130) = {math.log(130):.4f} (lech {abs(4.9122-math.log(130)):.3f} < 0.3 -> hop ly)")
    # 5d OOV: emoji + ky tu ngoai vocab
    oov_prompt = "Chào bạn 🙂 €"
    dropped = [c for c in f"<user> {oov_prompt}\n<bot> " if c not in stoi]
    out_oov = run_answer(model, stoi, itos, oov_prompt, 0.3, 9101)
    sanity["oov_dropped_chars"] = dropped
    sanity["oov_output"] = out_oov
    out_lines.append(f"OOV prompt {oov_prompt!r}: ky tu bi bo lang le = {dropped} -> OUTPUT: {out_oov!r} (khong crash)")
    # 5e prompt dai > block_size 128
    long_prompt = "Chào bạn, " * 25 + "bạn khỏe không?"
    out_long = run_answer(model, stoi, itos, long_prompt, 0.3, 9102)
    sanity["long_prompt_len"] = len(f"<user> {long_prompt}\n<bot> ")
    sanity["long_output"] = out_long
    out_lines.append(f"prompt dai {sanity['long_prompt_len']} ky tu (> block_size 128): OUTPUT: {out_long!r} (khong crash)")
    # 5f prompt 31 nhat quan: so tuoi o 2 temp o tren
    a31 = [r for r in results["prompts"] if r["id"] == 31]
    sanity["age_outputs"] = {str(r["temp"]): r["output"] for r in a31}
    ages = {str(r["temp"]): re.findall(r"\d+", r["output"]) for r in a31}
    sanity["age_numbers"] = ages
    out_lines.append(f"prompt [31] tuoi: so trich duoc theo temp = {ages}")
    results["sanity"] = sanity

    with open(os.path.join(HERE, "debug-round2-outputs.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    with open("/tmp/debug_probe_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # tom tat console
    for temp in TEMPS:
        rows = [r for r in results["prompts"] if r["temp"] == temp]
        from collections import Counter
        print(f"temp={temp}:", Counter(r["score"] for r in rows))
    for temp in (0.3, 0.9):
        rows = [r for r in results["math"] if r["temp"] == temp]
        print(f"math temp={temp}: dung {sum(r['ok'] for r in rows)}/{len(rows)}; "
              f"phan bo pred: {sorted(__import__('collections').Counter(r['pred'] for r in rows).items(), key=lambda x: -x[1])}")
    print("bye:", [ (r['q'], r['verdict']) for r in results["bye"]])
    for name in ("in_corpus", "variant"):
        rows = memo[name]
        print(f"memo {name}: exact_bot {sum(r['exact_bot'] for r in rows)}/{len(rows)}, "
              f"verbatim>=30: {sum(r['verbatim_run']>=30 for r in rows)}/{len(rows)}, "
              f"verbatim_run trung binh {sum(r['verbatim_run'] for r in rows)/len(rows):.1f}")
    print("sanity:", sanity)


if __name__ == "__main__":
    main()
