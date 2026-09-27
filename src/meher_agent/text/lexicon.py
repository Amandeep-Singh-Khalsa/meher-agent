"""Synonym clusters and stopwords for retrieval.

One dictionary, ``SYNONYMS``, maps a canonical lowercase key to every surface
form that should retrieve the same thing: English, romanised Hindi and
Devanagari. At import time each surface form is folded to Latin, casefolded and
split into tokens, and the tokens are merged with a union-find, so a token that
appears in two clusters (``laddoo`` belongs to both the motichoor and the besan
cluster) expands to the whole family rather than to one arbitrary key.

Product clusters here are the source of the folding gaps: ``fold_devanagari``
renders "काजू कटली" as "kaju katali", so that exact Latin spelling lives in this
cluster and a Devanagari query tokenises straight into it.
"""

from __future__ import annotations

import re

from .translit import fold_devanagari

__all__ = ["SYNONYMS", "STOPWORDS", "expand", "expand_all", "tokenize"]

#: Letters and numbers, with a decimal point kept inside its number: "2.5" is one
#: token, not two. A decimal is dropped when it does not sit between digits, so
#: "Rs.500" and "10.30 pm" still read as "rs"/"500" and "10.30"/"pm".
_TOKEN_SPLIT = re.compile(r"\d+(?:\.\d+)+|[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Fold to Latin, casefold, and split on everything that is not a letter or digit.

    Digits survive tokenisation because the resolver reads quantities out of them.
    """
    if not text:
        return []
    return _TOKEN_SPLIT.findall(fold_devanagari(text).casefold())


# --------------------------------------------------------------------------
# Synonym clusters
# --------------------------------------------------------------------------

SYNONYMS: dict[str, tuple[str, ...]] = {
    # -- commercial questions ---------------------------------------------
    "price": (
        "price", "prices", "rate", "rates", "cost", "costing", "pricing", "mrp",
        "dam", "daam", "dama", "kitna", "kitne", "kitni", "kitana", "kitne",
        "kimat", "kimai", "moolya", "mulya", "muvalya", "paisa", "paise",
        "paisa lagta", "kitna padega", "kitna lagega", "kitna hoga", "kitna padegi",
        "rate batao", "cost batao", "price list", "rate list", "cost list",
        "प्राइस", "प्राईस", "मूल्य", "मूल्य जानना", "दाम", "दाम बताइए", "कीमत",
        "कितना", "कितने", "पैसा", "पैसे",
    ),
    "delivery": (
        "delivery", "deliver", "delivered", "delivering", "dispatch", "drop",
        "drop off", "send", "send it", "shipping", "ship", "courier", "reach",
        "arrive", "arrival", "same day delivery", "home delivery", "doorstep",
        "bhejo", "bhej", "bhejiye", "pahunchao", "pahunchana", "pahunchega",
        "डिलीवरी", "डिलवरी", "डिलीवर", "भेजना", "भेजो", "पहुंचाना", "पहुंचाओ",
        "पहुँचाना", "भेजिए", "डिलीवरी करते",
    ),
    "distance": (
        "distance", "km", "kilo", "kilometer", "kilometre", "kilometers",
        "kilometres", "kilomitar", "kilomitra", "kilometer door", "door",
        "door se", "away", "se door", "kitne km", "how far", "radius",
        "दूर", "दूरी", "किलोमीटर", "किलोमीटर दूर", "दूर है", "कितने किलोमीटर",
    ),
    "discount": (
        "discount", "discounts", "off", "offer", "offers", "sale", "deal",
        "concession", "rebate", "coupon", "voucher", "promo", "offered",
        "free", "cheaper", "cheap", "sasta", "saasta", "bachat", "chhoot",
        "chut", "percent off", "% off", "extra off", "morediscount",
        "छूट", "छुट", "डिस्काउंट", "डिसकाउंट", "ऑफर", "सस्ता", "रिबेट",
        "कॉउपन", "छूट मिलेगी", "डिस्काउंट मिलेगा",
    ),
    "order": (
        "order", "orders", "booking", "book", "reserve", "reservation",
        "pre order", "preorder", "place", "placement", "cart", "basket",
        "mango", "mangwao", "mangwayo", "mangwana", "mangwaiye", "karwaye",
        "bhejo", "order karna", "order karo", "order kardo", "new order",
        "ऑर्डर", "आर्डर", "मंगाना", "मंगवाना", "मंगवाओ", "मंगाओ", "ऑर्डर करना",
    ),
    "return": (
        "return", "returned", "returning", "refund", "refunded", "replacement",
        "replace", "money back", "wapas", "vapas", "wapas", "return karo",
        "wapas karna", "exchange", "exchange karo", "paisa wapas",
        "वापस", "वापिस", "रिफंड", "रिफंड मिलेगा", "वापस करना", "बदलवाना",
    ),
    "damaged": (
        "damage", "damaged", "broken", "crushed", "spoiled", "spoilt", "rotten",
        "melted", "leaking", "torn", "bent", "messy", "wrong item",
        "kharab", "kharab hai", "toota", "tooti", "thela", "galat",
        "खराब", "टूटा", "टूटी", "मैला", "गलत", "सड़ा",
    ),
    "complaint": (
        "complaint", "complaints", "complain", "unhappy", "disappointed",
        "disappointing", "not happy", "bad service", "poor service", "rude",
        "shocked", "pathetic", "furious", "angry", "shikayat", "shikayati",
        "complain karo", "complaint karna", "shaam",
        "शिकायत", "शिकायत करनी", "नाराज", "असंतुष्ट",
    ),
    "photo": (
        "photo", "photos", "picture", "pictures", "image", "snapshot", "pic",
        "evidence", "video", "forwarded", "forward",
        "फोटो", "तस्वीर", "फोटो भेजा",
    ),
    "bulk": (
        "bulk", "bulky", "wholesale", "whole sale", "party order", "corporate",
        "office order", "bhandara", "function", "event", "large quantity",
        "big order", "bada order", "badi quantity", "thoda sa zyada", "lot",
        "बड़ा ऑर्डर", "थोक", "ढेर", "बैंक्वेट",
    ),
    "wedding": (
        "wedding", "wedding order", "custom", "custom order", "customised",
        "customized", "bespoke", "catering", "cake", "cake order", "pastry",
        "barat", "shaadi", "sagan", "mehndi", "haldi", "reception",
        "dowry", "thali", "ब्याह", "शादी", "कस्म", "कैक", "कस्टम", "थाली",
        "मिठाई का ऑर्डर", "वेडिंग",
    ),
    "advance": (
        "advance", "advance payment", "advance amount", "prepayment",
        "pre payment", "part payment", "token amount", "advance dena",
        "अग्रिम", "अग्रिम भुगतान", "पहले से भुगतान", "एडवांस",
    ),
    "notice": (
        "notice", "notice period", "in advance how many days", "how much notice",
        "lead time", "intimation", "suchna", "day ka notice", "puri tarah se",
        "पहले से बताना", "सूचना", "कितने दिन पहले",
    ),
    "payment": (
        "payment", "pay", "paid", "paying", "upi", "neft", "net banking",
        "bank transfer", "card", "debit card", "credit card", "visa", "mastercard",
        "rupay", "cod", "cash on delivery", "cash", "money", "bhejna hai",
        "payment kaise", "payment karu", "payment karo", "kaidun", "kaidun kare",
        "भुगतान", "पैसे भेजने", "नकद", "कार्ड", "उपाय", "यूपीआई",
    ),
    "gst": (
        "gst", "gstin", "gst included", "tax", "tax included", "vat",
        "invoice", "bill", "bill banega", "invoice milega", "tax bill",
        "receipt", "bill dedo", "GST", "जीएसटी", "इनवॉइस", "बिल", "कर",
        "टैक्स", "जी एस टी",
    ),
    "allergen": (
        "allergen", "allergens", "allergy", "allergic", "allergies", "nut",
        "nuts", "nut allergy", "peanut", "peanuts", "cashew", "almond",
        "almonds", "milk", "dairy", "wheat", "gluten", "soy", "trace",
        "traces", "veg", "vegetarian", "non veg", "nonveg", "egg", "eggs",
        "eggless", "puran veg", "shaahi", "kaju allergy", "has cashew",
        "contains nuts", "safe for",
        "एलर्जी", "एलर्जीक", "काजू", "पीनट", "बादाम", "दूध", "गेहूँ", "सोयाबीन",
        "शुद्ध शाकाहारी", "अंडा", "अंडे", "वीग", "नॉन वीग",
    ),
    "storage": (
        "storage", "store", "fridge", "refrigerator", "refrigerate", "refrigerated",
        "shelf life", "shelf", "expiry", "expiration", "expire", "expires",
        "how long", "how many days", "fresh for", "eat within", "best before",
        "use by", "room temperature", "accha rakh", "rakhne", "rakhna hai",
        "फ्रिज", "फ़्रिज", "शेल्फ लाइफ", "एक्सपायरी", "कब तक खाये",
        "कितने दिन", "सुरक्षित रख",
    ),
    "hours": (
        "hours", "hour", "timing", "timings", "time", "open", "opens", "opening",
        "close", "closes", "closing", "shop open", "abhi khula", "band",
        "khula hai", "band hai", "holi", "holiday", "closed", "shut",
        "today", "aaj", "timing kya", "kitne baje", "subah", "shaam", "raat",
        "समय", "खुला", "बंद", "खुला है", "बंद है", "आज", "छुट्टी", "टाइमिंग",
    ),
    "address": (
        "address", "location", "where", "directions", "direction", "near",
        "nearby", "landmark", "map", "shop address", "located", " situated",
        "kahan", "kahan hai", "kahan par", "pata", "pata batao", "rajouri",
        "central market", "new delhi", "pin code", "pincode", "postcode",
        "पता", "पता बताइए", "कहाँ", "कहां", "कहाँ है", "पता बताओ", "निकट",
        "मार्केट", "पिन कोड",
    ),
    "contact": (
        "contact", "phone", "call", "mobile", "number", "whatsapp", "whats app",
        "email", "mail", "owner", "manager", "staff", "bhejo number", "reach you",
        "phone number", "call karna", "email id", "reach out", "connect",
        "संपर्क", "फोन", "फ़ोन", "नंबर", "मोबाइल", "व्हाट्सएप", "ईमेल",
        "मेल आईडी", "मालिक", "फोन नंबर",
    ),
    "how_to_order": (
        "how to order", "how do i order", "order kaise kare", "order kaise karu",
        "order process", "ordering process", "kaise order", "order karne ka",
        "ऑर्डर कैसे", "कैसे करें", "कैसे करे", "ऑर्डर करने का तरीका",
    ),
    "pickup": (
        "pickup", "pick up", "takeaway", "take away", "collect", "collection",
        "counter", "self pickup", "udhar lena", "uthana", "lena hai",
        "पिकअप", "उठाना", "स्वयं ले जाऊंगा",
    ),
    "language": (
        "language", "languages", "hindi", "hinglish", "english", "bhasha",
        "which language", "kisme baat", "bata sakte", "Hindi me",
        "भाषा", "हिंदी", "अंग्रेजी", "किस भाषा",
    ),
    "fresh": (
        "fresh", "freshly made", "made today", "made daily", "freshly prepared",
        "today morning", "stale", "old stock", "bzati hui", "nayi", "taaza",
        "ताजा", "आज बनी", "ताज़ा", "फ्रेश", "नहीं बनी",
    ),
    "sweet": (
        "sweet", "sweets", "mithai", "mithai ka", "indian sweet", "halwai",
        "मिठाई", "मिठाई का", "भारतीय मिठाई", "हलवाई", "मीठा", "मिठास",
    ),
    "snack": (
        "snack", "snacks", "savory", "savoury", "namkeen ka", "tea time",
        "नमकीन", "नाश्ता", "स्नैक",
    ),
    "weight": (
        "weight", "kilo gram", "kilogram", "gram", "grams", "gm",
        "quantity", "qty", "wa weigh", "vajan", "half kg", "pack", "packs",
        "packet", "वजन", "वज़न", "किलो", "किलोग्राम", "ग्राम", "मात्रा", "पैकेट",
        "पैकिंग", "कितने किलो",
    ),
    "piece": (
        "piece", "pieces", "pc", "pcs", "nos", "nos piece", "each", "per piece",
        "unit", "units", "single", "one piece", "dozen", "a dozen", "couple",
        "a couple", "pair", "half dozen", "बीस", "दोजन", "नग",
    ),
    "box": (
        "box", "boxes", "packing box", "gift box", "giftbox", "gifth box",
        "hamper", "gift hamper", "hamper box", "gift basket", "basket",
        "combo", "combo box", "boks", "boks", "boks", "boxs", "बक्सा", "बॉक्स",
        "गिफ्ट बॉक्स", "गिफ्ट बॉक्स", "उपहार", "टोकरी", "गिफ्ट", "पैकिंग बॉक्स",
    ),
    # -- the fourteen catalog items ---------------------------------------
    "kaju_katli": (
        "kaju katli", "kaju katli", "kaju katly", "kaju katli sweet", "kaju",
        "katli", "katly", "kaju kathli", "kaju katali", "kaju katlee",
        "kajur katli",
        "काजू कटली", "काजू कटली", "काजू", "कटली", "काजू कतली",
    ),
    "sugar_free_kaju_katli": (
        "sugar free kaju katli", "sugarfree kaju katli", "sugar-free kaju katli",
        "kaju katli sugar free", "sugar free kaju", "sugar free katli",
        "sugar free", "sugarfree", "sugar-free", "no sugar kaju katli",
        "diet kaju katli", "kksf", "sugarless",
        "शुगर फ्री काजू कटली", "सुगर फ्री काजू कटली", "शुगर फ्री", "सुगर फ्री",
        "चीनी मुक्त काजू कटली", "बिना चीनी",
    ),
    "motichoor_laddoo": (
        "motichoor laddoo", "motichoor ladoo", "motichur laddoo", "moti laddoo",
        "moti ladoo", "moti laddu", "motichoor", "moti", "moori", "moti boondi",
        "pearl laddoo", "boondi laddoo", "मोतीचूर लड्डू", "मोतीचोर लड्डू",
        "मोतीचूर", "मोतीचोर", "मोती लड्डू", "मुखी लड्डू",
    ),
    "besan_laddoo": (
        "besan laddoo", "besan ladoo", "besan laddu", "besan laddus", "besan",
        "basan laddoo", "besan ke laddoo", "gram flour laddoo",
        "बेसन लड्डू", "बेसन लड्डू", "बेसन", "आटे के लड्डू",
    ),
    "laddoo": (
        "laddoo", "laddo", "ladoo", "laddu", "laddus", "laddus", "laddoos",
        "लड्डू", "लड्डू", "लड्डु", "लद्दू", "स्वादिष्ट लड्डू",
    ),
    "soan_papdi": (
        "soan papdi", "soan papdi", "son papdi", "soan papri", "soan papada",
        "son papadi", "papdi", "papri", "papadi", "soan papde",
        "सोन पापड़ी", "सोन पापड़ी", "सोन पापड़ी", "सोन पापडी", "पापड़ी", "पापड़ी",
        "स्वेट पापड़ी",
    ),
    "gulab_jamun": (
        "gulab jamun", "gulab jamun", "golab jamun", "gulab jamoons", "jamun",
        "jamoons", "gulab", "gulaab jamun", "gulab jamun sweet",
        "गुलाब जामुन", "गुलाब जामुन", "जामुन", "गुलाब जामुन मिठाई",
    ),
    "rasmalai": (
        "rasmalai", "ras malai", "rasa malai", "rasmalai", "rasmlai", "rasmai",
        "malai", "kulfi rasmalai",
        "रसमलाई", "रसमलाई", "रस मलाई", "रसमलाई का",
    ),
    "mixed_namkeen": (
        "mixed namkeen", "namkeen", "namkean", "mix namkeen", "mixed snack",
        "mixture namkeen", "नमकीन", "नमकीन", "मिक्स नमकीन", "मिश्रित नमकीन",
    ),
    "aloo_bhujia": (
        "aloo bhujia", "aloo bhujia", "alu bhujia", "aloo bhujiya", "aloo bhujiya",
        "bhujia", "bhujiya", "aloo bhujia namkeen", "आलू भुजिया", "आलू भुजिया",
        "भुजिया", "आलू का भुजिया",
    ),
    "samosa": (
        "samosa", "samosa", "samosas", "samosay", "samoosa", "samoose", "singara",
        "समोसा", "समोसा", "समोसे", "सिंघाड़ा",
    ),
    "dhokla": (
        "khaman dhokla", "dhokla", "dhokla", "khaman", "dhokle", "khaman dhokle",
        "handvo", "dhokla namkeen",
        "खमन धोकला", "धोकला", "धोकले", "खमन", "हांडवो",
    ),
    "gift_box_small": (
        "diwali gift box small", "small gift box", "small box", "small diwali box",
        "small diwali gift box", "chhota gift box", "chota gift box",
        "chhoti gift box", "mini gift box", "gbs", "small hamper",
        "छोटा गिफ्ट बॉक्स", "छोटे गिफ्ट बॉक्स", "गिफ्ट बॉक्स छोटा",
        "छोटा डिब्बा",
    ),
    "gift_box_large": (
        "diwali gift box large", "large gift box", "big gift box", "bada gift box",
        "badi gift box", "bade gift box", "bada diwali box", "badi diwali box",
        "large diwali box", "big diwali box", "premium gift box",
        "dry fruit gift box", "gbl", "large hamper", "big box", "bada box",
        "बड़ा गिफ्ट बॉक्स", "बड़े गिफ्ट बॉक्स", "गिफ्ट बॉक्स बड़ा", "बड़ा डिब्बा",
        "प्रीमियम गिफ्ट बॉक्स",
    ),
}

#: Function words, dropped before BM25 scoring. English plus transliterated Hindi.
#: Deliberately excludes "bada"/"chhota": they are the only size qualifiers that
#: distinguish the two gift boxes, and they must survive into the SKU index.
STOPWORDS: frozenset[str] = frozenset(
    tokenize(
        """
        a an the and or but of to in on at for from with without by as is are was
        were be been being am do does did doing done i me my mine myself you your
        yours yourself he she it we us our ours they them their theirs this that
        these those what which who whom whose when where why how please thanks
        thank thankyou hi hello hey ok okay yes yeah yep no nope sure fine good
        great nice want wants wanting need needs needed would could will shall may
        might must have has had having there here also too very just so if then
        more most much many some any all each every other another same
        me us get got getting make makes made making take takes taken give gives
        given add adds added adding let lets put puts about into over under out
        up down off now new old first second next last one ones thing things
        lot lots bit little can cannot cant dont doesnt didnt wont cant
        sir madam please kindly regards hi hello thanks thank you bye bye
        अंग्रेजी अंग्रेज़ी में है हैं हो होता होती होते था थी थे यह वह ये वे से को के का
        की के लिए लिये और या तो ही भी नहीं ना ने पर तक जी मैं हम आप तुम मेरे मेरी
        आपका आपकी तुम्हारा हमारा मुझे मुझको बताओ बताइए बताइये बताना बताया बताता
        कितना कितने कितनी क्या क्यों कैसे कहाँ कब कौन सब बहुत थोड़ा ज़्यादा ज्यादा
        एक दो तीन चार पांच पाँच छह छः सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह
        सोलह सत्रह अठारह उन्नीस बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस
        अट्ठाईस उनतीस तीस हजार लाख करोड़ रुपये रुपए रु पैसे पैसा
        दूर पास यहाँ वहाँ अब अभी जल्दी बाद पहले साथ ऊपर नीचे अंदर बाहर
        दिन रात सुबह शाम दोपहर कल परसों आज
        तो भी कि की के और फिर सकता सकते करना करने करें करो कर दो लो ले देना देने
        चाहता चाहती चाहते चाहिए चाहिये मिलेगा मिलेगी मिले हुआ हुई हुए रहा रही रहे
        कृपया बस सिर्फ
        """,
    )
) | frozenset(
    {
        "hai", "hain", "ho", "hoon", "hu", "hun", "kar", "karo", "karna", "karte",
        "karke", "kardo", "karenge", "ka", "ki", "ke", "ko", "se", "aur", "ya",
        "ek", "do", "tin", "char", "paanch", "chhe", "saat", "aath", "nau", "das",
        "bhaiya", "bhai", "yaar", "aap", "tum", "mera", "meri", "hamara", "kripya",
        "namaste", "dhanyavaad", "ab", "abhi", "jaldi", "waqt", "kal", "parso",
        "aaj", "kitna", "kitne", "kitni", "kya", "kyun", "kyun", "kahan", "kaise",
        "kaisa", "kaisi", "chahiye", "chaiye", "batao", "bata", "bataiye", "dena",
        "deno", "de", "dene", "dijiye", "lelo", "lena", "mangwao", "mangwayo",
        "mangwana", "bhejo", "bhej", "dekhlo", "dekhna", "theek", "accha", "acha",
        "aur", "phir", "fir", "bas", "sirf", "hi", "bhi", "tak", "waala", "wala",
        "wale", "wali", "gali", "ji", "jii", "jiji", "uncle", "aunty", "didi",
        "zaroor", "zaroori", "chalo", "chalega", "chalegi", "shukriya", "paisa",
        "paise", "rupaiya", "rupaye", "thoda", "thodi", "zyada", "bahut", "bohot",
        "abka", "aajka", "kahin", "wahan", "yahan", "idhar", "udhar", "sabko",
        "sab", "koi", "kuch", "nahi", "nahin", "haan", "han", "milonga", "milenge",
        "laga", "lagta", "padega", "hoga", "denge", "dunga", "dungi",
    }
)


# --------------------------------------------------------------------------
# Cluster index
# --------------------------------------------------------------------------

#: Function words and auxiliaries. They may appear inside a cluster's surface
#: forms, but they must not drag every cluster that mentions them into one
#: family: "hai" occurs in the hours, address and damaged forms, and a transitive
#: union over it collapses the whole dictionary. A non-content token therefore
#: resolves to the *first* cluster that declares it, which is why "kitna" lands
#: on "price" rather than on price + distance + storage + weight + notice.
_GRAMMAR: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at",
        "for", "from", "with", "by", "as", "is", "are", "was", "were", "be",
        "do", "does", "did", "i", "me", "my", "you", "your", "it", "we", "us",
        "how", "what", "when", "where", "which", "who", "why", "please",
        "can", "could", "would", "will", "ka", "ki", "ke", "ko", "se", "mein",
        "par", "aur", "ya", "tak", "bhi", "hi", "hai", "hain", "ho", "hu",
        "hoon", "kya", "kyun", "kyu", "kahan", "kaise", "kaisa", "kaisi",
        "kitna", "kitne", "kitni", "kitana", "kitane", "karna",
        "karte", "karke", "kar", "karo", "kardo", "dene", "dena", "deno",
        "de", "dedo", "dede", "lena", "lelo", "lo", "mango", "mangwao",
        "mangwayo", "mangwana", "mangwaiye", "bhejo", "bhej", "bhejiye",
        "dekhlo", "dekhna", "dekh", "batao", "bata", "bataiye", "bataye",
        "chahiye", "chaiye", "milega", "milegi", "milenge", "milaega",
        "padega", "padegi", "lagega", "lagta", "hoga", "hogi", "hua", "hui",
        "bataie", "batayie", "bataye", "bataijiye", "batana", "batati",
        "hone", "rakhna", "rakh", "bana", "banana", "banwao", "chuna",
        "chuno", "khana", "peena", "piyo", "aao", "aana", "lagao", "laga",
        "aap", "aapka", "aapki", "aapke", "aapko", "tum", "tumhe", "mera",
        "meri", "hamara", "hamari", "mujhe", "mujhko", "bhaiya", "bhai",
        "yaar", "didi", "kripya", "krupya", "namaste", "ab", "abhi", "jaldi",
        "waqt", "kal", "parso", "aaj", "raat", "shaam", "subah", "ghar",
        "yahan", "wahan", "idhar", "udhar", "phir", "fir", "bas", "sirf",
        "bahut", "thoda", "thodi", "zyada", "zaroor", "shukriya", "dhanyavaad",
        "koi", "kuch", "nahi", "nahin", "haan", "han", "abka", "aajka", "sab",
        "sabko", "wala", "wale", "wali", "waala", "kaun", "uncle", "aunty",
        "samay", "samaye", "kiss", "liye", "lijiye", "not", "per", "via",
        "such", "than", "then", "into", "onto", "upon", "very", "much", "many",
        "all", "any", "some", "more", "most", "each", "kripya",
    }
)

_CLUSTER_TOKENS: dict[str, frozenset[str]] = {
    key: frozenset(tok for form in forms for tok in tokenize(form))
    for key, forms in SYNONYMS.items()
}

#: Token -> every cluster key that declares it, in declaration order. A token two
#: clusters mention ("katli", by plain and by sugar-free katli) is ambiguous, so it
#: expands to the union of those families: a bare "katli" really can be either,
#: and it must still reach both spellings. The union stops there, at one hop, so
#: no family drags in a family that never mentioned the token.
_DECLARED_BY: dict[str, list[str]] = {}
for _key, _tokens in _CLUSTER_TOKENS.items():
    for _token in _tokens:
        _DECLARED_BY.setdefault(_token, []).append(_key)

#: Token -> canonical cluster. Diagnostic only.
CANONICAL: dict[str, str] = {
    token: keys[0] for token, keys in _DECLARED_BY.items()
}


def _payload(keys: list[str]) -> set[str]:
    out: set[str] = set()
    for key in keys:
        out |= _CLUSTER_TOKENS[key]
    return out - _GRAMMAR - STOPWORDS


def expand(token: str) -> set[str]:
    """The token plus every surface form that should retrieve the same thing.

    Devanagari is folded first, because the retrieval indexes are Latin. An
    unknown token comes back as its folded, casefolded self.
    """
    folded = fold_devanagari(token or "").casefold().strip()
    if not folded:
        return set()
    keys = _DECLARED_BY.get(folded)
    if keys is not None:
        return {folded} | _payload(keys)
    return {folded}


def expand_all(tokens: list[str]) -> list[str]:
    """``expand`` over a whole token list, order-preserved and de-duplicated."""
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        for expanded in sorted(expand(token)):
            if expanded not in seen:
                seen.add(expanded)
                out.append(expanded)
    return out
