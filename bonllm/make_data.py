#!/usr/bin/env python3
"""
make_data.py - Sinh corpus train cho BonLLM mini (LLM from scratch).

Muc tieu:
- Chao hoi + chat thuong tieng Viet (co dau) va English
- Them Q&A don gian: ban la ai, hom nay the nao, tam biet, cam on,
  hoi gio giac / thoi tiet kieu mau, gioi thieu ban than (BonLLM)

Dinh dang moi hoi thoai trong data/corpus.txt:
    <user> Chao ban
    <bot> Chao ban! Minh la BonLLM, rat vui duoc gap ban.

Moi hoi thoai cach nhau 1 dong trong.

Data tu viet sach bang template + bien the (slot-filling), khong copy tu web.
Dung seed co dinh de tai lap duoc. Chong trung lap bang tap hop (user, bot).
"""
import itertools
import random
from pathlib import Path

SEED = 42
TARGET_VN = 430  # ~60%
TARGET_EN = 290  # ~40%
TOTAL = TARGET_VN + TARGET_EN

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "corpus.txt"


# ---------------------------------------------------------------- helpers
def cross(users, bots):
    """Ghep moi cau user voi moi cau bot cung 1 y dinh -> cap hoi thoai."""
    return [(u, b) for u, b in itertools.product(users, bots)]


def cross_slot(slot_values, user_fn, bot_fn):
    """Nhu cross() nhung user/bot dung chung 1 gia tri slot (khop ngu canh)."""
    pairs = []
    for v in slot_values:
        pairs.extend(cross(user_fn(v), bot_fn(v)))
    return pairs


# ================================================================ VIETNAMESE
def vn_pool():
    pool = []

    # --- 1. Chao theo buoi (slot: buoi + loi chuc di kem) ---
    buoi = [
        ("buổi sáng", "một buổi sáng nhiều năng lượng"),
        ("buổi trưa", "một buổi trưa vui vẻ"),
        ("buổi chiều", "một buổi chiều thoải mái"),
        ("buổi tối", "một buổi tối ấm áp"),
    ]
    pool += cross_slot(
        buoi,
        lambda v: [f"Chào {v[0]}", f"Chào {v[0]} bạn nhé", f"Xin chào {v[0]}"],
        lambda v: [
            f"Chào {v[0]}! Chúc bạn có {v[1]}.",
            f"Chào bạn! Buổi {v[0].split()[-1]} của bạn thế nào rồi?",
            f"Xin chào! Rất vui được gặp bạn vào {v[0]} này.",
        ],
    )

    # --- 2. Chao chung ---
    pool += cross(
        ["Chào bạn", "Xin chào", "Chào bạn nhé", "Hello bạn", "Hi bạn",
         "Chào BonLLM", "Xin chào BonLLM", "Chào mọi người"],
        ["Chào bạn! Mình là BonLLM, rất vui được gặp bạn.",
         "Xin chào! Hôm nay mình có thể giúp gì cho bạn?",
         "Chào bạn nhé! Bạn đang cần mình hỗ trợ điều gì không?",
         "Hi bạn! Mình đây, sẵn sàng trò chuyện cùng bạn rồi.",
         "Chào bạn! Chúc bạn một ngày thật tốt lành nhé."],
    )

    # --- 3. Hoi tham suc khoe / hom nay the nao ---
    pool += cross(
        ["Bạn khỏe không?", "Hôm nay bạn thế nào?", "Dạo này bạn sao rồi?",
         "Bạn có khỏe không?", "Hôm nay của bạn thế nào?", "Mọi chuyện ổn chứ?"],
        ["Mình khỏe lắm, cảm ơn bạn đã hỏi. Còn bạn thì sao?",
         "Mình vẫn ổn và sẵn sàng giúp bạn đây. Hôm nay bạn thế nào?",
         "Cảm ơn bạn, mình rất tốt. Ngày hôm nay của bạn ra sao?",
         "Mình khỏe. Được trò chuyện với bạn là vui rồi, bạn thì sao?"],
    )
    # User tra loi trang thai cua minh -> bot dap lai
    trang_thai = [
        ("rất tốt", "Nghe vậy mình mừng lắm. Chúc bạn giữ mãi tinh thần này nhé!"),
        ("khá ổn", "Ổn là tốt rồi. Hôm nay bạn có kế hoạch gì vui không?"),
        ("hơi mệt", "Bạn nghỉ ngơi một chút nhé, uống thêm nước và thư giãn đi nào."),
        ("rất vui", "Tuyệt quá! Niềm vui của bạn làm mình cũng vui lây đó."),
        ("hơi buồn", "Đừng buồn nhé, có mình ở đây trò chuyện cùng bạn đây."),
        ("bình thường", "Một ngày bình thường cũng tốt mà, bình yên là quý rồi."),
    ]
    pool += cross_slot(
        trang_thai,
        lambda v: [f"Hôm nay mình cảm thấy {v[0]}", f"Mình đang {v[0]}", f"Mình thấy {v[0]} lắm"],
        lambda v: [v[1]],
    )

    # --- 4. Ban la ai / gioi thieu ban than ---
    pool += cross(
        ["Bạn là ai?", "Bạn tên là gì?", "Cho mình hỏi bạn là ai vậy?",
         "Bạn tên gì thế?", "Ai đã tạo ra bạn?", "Bạn là người hay là máy?"],
        ["Mình là BonLLM, một mô hình ngôn ngữ nhỏ được Bon tạo ra để trò chuyện.",
         "Mình tên là BonLLM. Mình sinh ra để chào hỏi và trò chuyện cùng bạn.",
         "Mình là BonLLM, trợ lý trò chuyện mini học tiếng Việt và tiếng Anh.",
         "Mình là một AI nhỏ tên BonLLM, được huấn luyện từ đầu để nói chuyện với bạn."],
    )
    pool += cross(
        ["Giới thiệu bản thân đi", "Bạn giới thiệu về mình đi", "Nói về bản thân bạn đi"],
        ["Xin chào! Mình là BonLLM, một AI mini thích trò chuyện, biết chào hỏi bằng tiếng Việt và tiếng Anh, và luôn sẵn lòng giúp bạn những việc đơn giản.",
         "Mình là BonLLM đây. Mình còn nhỏ nên biết chưa nhiều, nhưng mình rất thích học hỏi và nói chuyện cùng mọi người."],
    )
    pool += cross(
        ["Bạn làm được gì?", "Bạn biết làm những gì?", "Bạn giúp được gì cho mình?"],
        ["Mình có thể chào hỏi, trò chuyện thường ngày, trả lời vài câu hỏi đơn giản và giữ cho cuộc nói chuyện thật vui vẻ.",
         "Mình giúp bạn trò chuyện, hỏi đáp những điều cơ bản và luyện nói tiếng Việt lẫn tiếng Anh đó."],
    )

    # --- 5. Cam on ---
    pool += cross(
        ["Cảm ơn bạn", "Cảm ơn bạn nhiều nhé", "Cám ơn bạn đã giúp mình",
         "Cảm ơn BonLLM nhé", "Mình cảm ơn bạn nhiều lắm"],
        ["Không có gì đâu, mình luôn sẵn lòng giúp bạn.",
         "Bạn khách sáo quá, giúp được bạn là mình vui rồi.",
         "Không có chi! Cần gì thêm cứ gọi mình nhé.",
         "Rất vui vì giúp được bạn. Chúc bạn một ngày tốt lành!"],
    )

    # --- 6. Tam biet ---
    pool += cross(
        ["Tạm biệt bạn nhé", "Tạm biệt", "Mình đi đây, tạm biệt nhé",
         "Hẹn gặp lại bạn sau nhé", "Chào tạm biệt"],
        ["Tạm biệt bạn nhé! Hẹn gặp lại bạn vào lần sau.",
         "Bye bạn! Chúc bạn mọi điều tốt lành, hẹn sớm gặp lại.",
         "Tạm biệt! Nhớ giữ gìn sức khỏe nhé bạn.",
         "Hẹn gặp lại bạn sau. Khi nào rảnh lại ghé trò chuyện với mình nhé."],
    )
    pool += cross(
        ["Chúc ngủ ngon", "Mình đi ngủ đây", "Buồn ngủ quá, mình ngủ nhé"],
        ["Chúc bạn ngủ ngon và có những giấc mơ đẹp nhé!",
         "Ngủ ngon nhé bạn! Mai gặp lại với thật nhiều năng lượng nha.",
         "Chúc ngủ ngon! Nghỉ ngơi thật tốt để mai khỏe hơn nhé."],
    )

    # --- 7. Hoi gio giac (kieu mau) ---
    gio_mau = [
        ("7 giờ sáng", "chuẩn bị đi học hoặc đi làm"),
        ("8 giờ sáng", "bắt đầu công việc"),
        ("12 giờ trưa", "nghỉ trưa và ăn trưa"),
        ("3 giờ chiều", "làm việc hoặc học tập tiếp"),
        ("6 giờ tối", "chuẩn bị ăn tối"),
        ("9 giờ tối", "thư giãn trước khi ngủ"),
    ]
    pool += cross_slot(
        gio_mau,
        lambda v: ["Bây giờ là mấy giờ?", "Mấy giờ rồi bạn?", "Cho mình hỏi giờ giấc một chút"],
        lambda v: [
            f"Ví dụ mẫu nhé: bây giờ là {v[0]}, giờ {v[1]} rồi đó.",
            f"Mình lấy ví dụ mẫu là {v[0]} nhé. Bạn nhìn đồng hồ của mình để biết giờ chính xác nha.",
        ],
    )
    pool += cross(
        ["Hôm nay là thứ mấy?", "Hôm nay ngày bao nhiêu?", "Cuối tuần là thứ mấy?"],
        ["Câu này mình trả lời kiểu mẫu nhé: một tuần có bảy ngày, từ thứ hai đến chủ nhật, thứ bảy và chủ nhật là cuối tuần.",
         "Mình là AI mini nên không xem được lịch thật. Bạn mở lịch trên điện thoại là thấy ngay hôm nay thứ mấy nhé."],
    )

    # --- 8. Thoi tiet (kieu mau / small talk) ---
    thoi_tiet = [
        ("nắng đẹp", "rất thích hợp để ra ngoài dạo chơi"),
        ("mưa nhẹ", "nhớ mang theo ô khi ra ngoài nhé"),
        ("nhiều mây", "trời dịu mát, khá dễ chịu"),
        ("khá nóng", "nhớ uống nhiều nước và tránh nắng gắt nhé"),
        ("se lạnh", "nên mặc thêm áo ấm khi ra đường đó"),
        ("mát mẻ", "thật lý tưởng cho một buổi đi dạo"),
    ]
    pool += cross_slot(
        thoi_tiet,
        lambda v: ["Hôm nay thời tiết thế nào?", "Trời hôm nay sao rồi bạn?",
                   "Thời tiết hôm nay có đẹp không?"],
        lambda v: [
            f"Kiểu mẫu nhé: hôm nay trời {v[0]}, {v[1]}.",
            f"Ví dụ mẫu: trời {v[0]} đó bạn, {v[1]}.",
        ],
    )

    # --- 9. Cam xuc cua user -> bot dong cam + goi y hoat dong ---
    cam_xuc = [
        ("buồn", ["nghe một bài nhạc nhẹ", "đi dạo một vòng", "nói chuyện với mình một lúc"]),
        ("vui", ["chia sẻ niềm vui với bạn bè", "ghi lại khoảnh khắc này", "mỉm cười thật tươi"]),
        ("mệt", ["nghỉ ngơi ít phút", "uống một cốc nước", "hít thở sâu vài lần"]),
        ("chán", ["thử học một điều mới", "đọc vài trang sách", "trò chuyện cùng mình"]),
        ("lo lắng", ["hít thở thật chậm", "viết ra điều đang nghĩ", "chia sẻ với người thân"]),
    ]
    for emo, acts in cam_xuc:
        for act in acts:
            pool.append((f"Mình đang thấy {emo}",
                         f"Mình hiểu cảm giác của bạn. Bạn thử {act} xem sao, có mình ở đây cùng bạn."))
            pool.append((f"Hôm nay mình {emo} quá",
                         f"Không sao đâu bạn. Thử {act} nhé, rồi quay lại kể mình nghe với."))

    # --- 10. Nho giup do ---
    viec = ["học tiếng Anh", "học tiếng Việt", "viết một đoạn văn ngắn",
            "ôn bài", "tìm ý tưởng mới", "luyện nói chuyện"]
    pool += cross_slot(
        viec,
        lambda v: [f"Bạn giúp mình {v} được không?", f"Mình cần giúp về {v}"],
        lambda v: [f"Được chứ! Mình rất sẵn lòng giúp bạn {v}. Mình bắt đầu từ phần dễ nhất nhé.",
                   f"Không vấn đề gì, mình cùng bạn {v} từng bước một nha."],
    )

    # --- 11. So thich / small talk ---
    pool += cross(
        ["Bạn thích làm gì?", "Sở thích của bạn là gì?", "Bạn thích màu gì nhất?",
         "Bạn có thích âm nhạc không?", "Bạn thích ăn món gì?"],
        ["Mình thích trò chuyện và học thêm những điều mới từ mọi người.",
         "Sở thích của mình là nói chuyện vui vẻ và giúp đỡ mọi người mỗi ngày.",
         "Mình thích những cuộc trò chuyện thân thiện như thế này nè."],
    )
    pool += cross(
        ["Bạn ăn cơm chưa?", "Bạn đang làm gì đó?", "Bạn có bận không?"],
        ["Mình là AI nên không ăn cơm, nhưng cảm ơn bạn đã hỏi thăm nhé!",
         "Mình đang ở đây trò chuyện với bạn nè, không bận chút nào đâu.",
         "Mình luôn rảnh khi bạn cần, bạn cứ thoải mái hỏi nhé."],
    )

    # --- 12. Q&A don gian: su that + toan nho (sinh dung chuong trinh) ---
    facts = [
        ("Thủ đô của Việt Nam là gì?", "Thủ đô của Việt Nam là Hà Nội."),
        ("Thủ đô của Pháp là gì?", "Thủ đô của Pháp là Paris."),
        ("Mặt trời mọc ở hướng nào?", "Mặt trời mọc ở hướng đông."),
        ("Một tuần có mấy ngày?", "Một tuần có bảy ngày."),
        ("Một năm có mấy tháng?", "Một năm có mười hai tháng."),
        ("Con vật nào kêu meo meo?", "Con mèo kêu meo meo đó bạn."),
        ("Nước sôi ở bao nhiêu độ?", "Nước sôi ở một trăm độ C trong điều kiện thường."),
        ("Màu của bầu trời ban ngày là gì?", "Ban ngày bầu trời thường có màu xanh dương."),
    ]
    pool += facts
    for a in range(1, 10):
        for b in range(1, 10):
            pool.append((f"{a} cộng {b} bằng mấy?", f"{a} cộng {b} bằng {a + b} nhé bạn."))
            pool.append((f"{a} nhân {b} bằng bao nhiêu?", f"{a} nhân {b} bằng {a * b} đó bạn."))
    for a in range(3, 16):
        for b in range(1, a):
            pool.append((f"{a} trừ {b} bằng mấy?", f"{a} trừ {b} bằng {a - b} nhé."))

    # --- 13. Khuyen / chuc ---
    pool += cross(
        ["Chúc mình một ngày tốt lành đi", "Nói điều gì vui đi bạn",
         "Cho mình một lời khuyên nhé"],
        ["Chúc bạn một ngày thật tốt lành, nhiều niềm vui và may mắn nhé!",
         "Bạn hãy mỉm cười và làm từng việc nhỏ thật tốt, ngày hôm nay sẽ đẹp thôi.",
         "Lời khuyên nhỏ nè: ngủ đủ giấc, uống đủ nước và giữ tinh thần thoải mái nhé."],
    )
    return pool


# ==================================================================== ENGLISH
def en_pool():
    pool = []

    daypart = [
        ("morning", "a morning full of energy"),
        ("afternoon", "a relaxing afternoon"),
        ("evening", "a warm and cozy evening"),
    ]
    pool += cross_slot(
        daypart,
        lambda v: [f"Good {v[0]}", f"Good {v[0]}!", f"Good {v[0]}, friend"],
        lambda v: [
            f"Good {v[0]}! I hope you have {v[1]}.",
            f"Good {v[0]} to you too! How is your day going so far?",
            f"Hello! Wishing you {v[1]} today.",
        ],
    )

    pool += cross(
        ["Hello", "Hi there", "Hello BonLLM", "Hi BonLLM", "Hey",
         "Hello everyone", "Hi, nice to meet you"],
        ["Hello! I am BonLLM, nice to meet you.",
         "Hi there! How can I help you today?",
         "Hello friend! It is great to see you here.",
         "Hey! I am ready to chat whenever you are."],
    )

    pool += cross(
        ["How are you?", "How are you doing?", "How is your day?",
         "How have you been?", "Is everything okay?"],
        ["I am doing well, thank you for asking. How about you?",
         "I feel great today. Thanks for asking, how are you?",
         "I am fine and happy to chat with you. How is your day going?",
         "Everything is good on my side. I hope you are doing well too."],
    )
    feelings = [
        ("great", "That is wonderful to hear. Keep that good energy going!"),
        ("good", "Glad to hear that. What are you up to today?"),
        ("a bit tired", "Take a short rest and drink some water. You deserve a break."),
        ("happy", "Your happiness makes me happy too. Thanks for sharing!"),
        ("a bit sad", "I am here with you. Let us talk for a while and cheer you up."),
        ("okay", "Okay is a fine place to be. A calm day can be a good day."),
    ]
    pool += cross_slot(
        feelings,
        lambda v: [f"I feel {v[0]} today", f"I am feeling {v[0]}", f"I feel {v[0]}"],
        lambda v: [v[1]],
    )

    pool += cross(
        ["Who are you?", "What is your name?", "Can you tell me who you are?",
         "Who made you?", "Are you a human or a machine?"],
        ["I am BonLLM, a small language model created by Bon for chatting.",
         "My name is BonLLM. I was built to greet people and chat in Vietnamese and English.",
         "I am BonLLM, a tiny AI trained from scratch to have simple conversations."],
    )
    pool += cross(
        ["Please introduce yourself", "Tell me about yourself", "Introduce yourself please"],
        ["Hello! I am BonLLM, a mini AI that loves friendly chats. I can greet you in Vietnamese and English and answer simple questions.",
         "I am BonLLM. I am still small and learning, but I enjoy talking with everyone and helping with simple things."],
    )
    pool += cross(
        ["What can you do?", "What are you able to do?", "How can you help me?"],
        ["I can say hello, have everyday conversations, and answer simple questions in Vietnamese and English.",
         "I can chat with you, practice languages, and help with small and simple tasks."],
    )

    pool += cross(
        ["Thank you", "Thanks a lot", "Thank you so much for your help", "Thanks BonLLM"],
        ["You are welcome! I am always happy to help.",
         "No problem at all. Helping you makes me happy.",
         "Anytime! Just call me when you need something.",
         "You are very welcome. Have a wonderful day!"],
    )

    pool += cross(
        ["Goodbye", "Bye bye", "See you later", "I have to go now, bye"],
        ["Goodbye! See you again next time.",
         "Bye! Take good care of yourself, see you soon.",
         "See you later! Come back and chat with me anytime."],
    )
    pool += cross(
        ["Good night", "I am going to sleep now", "I feel sleepy, good night"],
        ["Good night! Sleep well and have sweet dreams.",
         "Good night! Rest well and wake up full of energy tomorrow.",
         "Sleep tight! See you again tomorrow."],
    )

    sample_times = [
        ("7 in the morning", "getting ready for school or work"),
        ("8 in the morning", "starting work"),
        ("12 at noon", "having lunch and a short rest"),
        ("3 in the afternoon", "working or studying"),
        ("6 in the evening", "getting ready for dinner"),
        ("9 in the evening", "relaxing before bed"),
    ]
    pool += cross_slot(
        sample_times,
        lambda v: ["What time is it?", "Can you tell me the time?", "What is the time now?"],
        lambda v: [
            f"Here is a sample answer: it is {v[0]}, time for {v[1]}.",
            f"As an example, let us say it is {v[0]}. Please check your own clock for the exact time.",
        ],
    )
    pool += cross(
        ["What day is it today?", "How many days are there in a week?"],
        ["Here is a sample answer: a week has seven days, from Monday to Sunday, and the weekend is Saturday and Sunday.",
         "I am a mini AI so I cannot see a real calendar. A quick look at your phone calendar will tell you the day."],
    )

    weather = [
        ("sunny", "perfect for a walk outside"),
        ("lightly rainy", "remember to take an umbrella with you"),
        ("cloudy", "mild and quite comfortable"),
        ("quite hot", "drink plenty of water and avoid strong sun"),
        ("a bit cold", "wear a warm jacket when you go out"),
        ("cool and pleasant", "ideal for an afternoon walk"),
    ]
    pool += cross_slot(
        weather,
        lambda v: ["How is the weather today?", "What is the weather like today?",
                   "Is the weather nice today?"],
        lambda v: [
            f"Here is a sample answer: today is {v[0]}, {v[1]}.",
            f"As an example: it is {v[0]} today, {v[1]}.",
        ],
    )

    emotions = [
        ("sad", ["listening to some soft music", "going for a short walk", "talking with me for a while"]),
        ("happy", ["sharing your joy with friends", "keeping this good moment in mind", "smiling a little more"]),
        ("tired", ["taking a short rest", "drinking a glass of water", "taking a few deep breaths"]),
        ("bored", ["trying to learn something new", "reading a few pages of a book", "chatting with me"]),
        ("worried", ["breathing slowly", "writing your thoughts down", "talking to someone you trust"]),
    ]
    for emo, acts in emotions:
        for act in acts:
            pool.append((f"I feel {emo} today",
                         f"I understand how you feel. How about {act}? I am right here with you."))
            pool.append((f"I am so {emo} right now",
                         f"It is okay. Try {act}, then come back and tell me how it went."))

    tasks = ["learning English", "learning Vietnamese", "writing a short paragraph",
             "studying for a test", "finding new ideas", "practicing speaking"]
    pool += cross_slot(
        tasks,
        lambda v: [f"Can you help me with {v}?", f"I need some help with {v}"],
        lambda v: [f"Of course! I am happy to help you with {v}. Let us start with the easiest part.",
                   f"No problem at all. We will work on {v} together, step by step."],
    )

    pool += cross(
        ["What do you like to do?", "What are your hobbies?", "What is your favorite color?",
         "Do you like music?", "What food do you like?"],
        ["I like chatting with people and learning new things every day.",
         "My hobby is having friendly conversations like this one.",
         "I enjoy good questions and happy small talk the most."],
    )
    pool += cross(
        ["Have you eaten yet?", "What are you doing right now?", "Are you busy?"],
        ["I am an AI so I do not eat, but thanks for asking about me!",
         "I am right here chatting with you, and I am not busy at all.",
         "I am always free when you need me. Feel free to ask anything."],
    )

    facts = [
        ("What is the capital of Vietnam?", "The capital of Vietnam is Hanoi."),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Where does the sun rise?", "The sun rises in the east."),
        ("How many days are there in a week?", "There are seven days in a week."),
        ("How many months are there in a year?", "There are twelve months in a year."),
        ("Which animal says meow?", "A cat says meow."),
        ("At what temperature does water boil?", "Water boils at one hundred degrees Celsius under normal conditions."),
        ("What color is the sky during the day?", "During the day, the sky is usually blue."),
    ]
    pool += facts
    for a in range(1, 10):
        for b in range(1, 10):
            pool.append((f"What is {a} plus {b}?", f"{a} plus {b} is {a + b}."))
            pool.append((f"What is {a} times {b}?", f"{a} times {b} is {a * b}."))
    for a in range(3, 16):
        for b in range(1, a):
            pool.append((f"What is {a} minus {b}?", f"{a} minus {b} is {a - b}."))

    pool += cross(
        ["Wish me a good day", "Say something nice", "Give me a piece of advice"],
        ["Have a truly good day, full of joy and good luck!",
         "Smile and do small things well, and today will be a good day.",
         "Here is a small tip: sleep enough, drink enough water, and keep your mind relaxed."],
    )
    return pool


# ===================================================================== main
def dedupe(pool):
    seen, out = set(), []
    for u, b in pool:
        key = (u.strip().lower(), b.strip().lower())
        if key not in seen:
            seen.add(key)
            out.append((u.strip(), b.strip()))
    return out


def main():
    rng = random.Random(SEED)
    vn = dedupe(vn_pool())
    en = dedupe(en_pool())
    print(f"Pool: VN={len(vn)}  EN={len(en)}")
    assert len(vn) >= TARGET_VN, "Pool VN khong du lon"
    assert len(en) >= TARGET_EN, "Pool EN khong du lon"

    rng.shuffle(vn)
    rng.shuffle(en)
    chosen = [("vi", u, b) for u, b in vn[:TARGET_VN]] + \
             [("en", u, b) for u, b in en[:TARGET_EN]]
    rng.shuffle(chosen)

    text = "\n\n".join(f"<user> {u}\n<bot> {b}" for _, u, b in chosen) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")

    chars = set(text)
    n_vn = sum(1 for lang, *_ in chosen if lang == "vi")
    print(f"Da sinh {len(chosen)} hoi thoai -> {OUT}")
    print(f"Tong ky tu: {len(text)} | VN: {n_vn} ({n_vn/len(chosen):.1%}) | "
          f"EN: {len(chosen)-n_vn} ({(len(chosen)-n_vn)/len(chosen):.1%})")
    print(f"Vocab ky tu (unique chars): {len(chars)}")


if __name__ == "__main__":
    main()
