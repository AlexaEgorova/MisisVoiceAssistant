from typing import Dict, List
from nltk.tokenize import word_tokenize

# Список команд
QUESTIONS: Dict[str, List[str]] = {
    "q_specializations": word_tokenize(
        "Какие направления подготовки есть в МИСИС".lower()
    ),
    "q_foundation": word_tokenize(
        "Расскажи, когда был основан МИСИС".lower()
    ),
    "q_year": word_tokenize(
        "В каком году был основан МИСИС".lower()
    ),
    "q_famous": word_tokenize(
        "Чем известен МИСИС".lower()
    ),
    "q_foreign": word_tokenize(
        "Много ли иностранных студентов в МИСИС".lower()
    ),
    "q_application": word_tokenize(
        "Как поступить в МИСИС".lower()
    ),
    "q_asp": word_tokenize(
        "Какие есть направления подготовки в аспирантуре".lower()
    ),
    "q_lab": word_tokenize(
        "Есть ли в МИСИС лаборатория".lower()
    ),
    "q_sertificate": word_tokenize(
        "Как получить справку".lower()
    ),
    "q_perspectives": word_tokenize(
        "Кем работают выпускники МИСИС".lower()
    ),
    "q_famous_people": word_tokenize(
        "Какие известные люди учились в МИСИС".lower()
    ),
    "q_military_department": word_tokenize(
        "Есть ли в МИСИС военная кафедра".lower()
    ),
    "q_double_diploma": word_tokenize(
        "Можно ли получить двойной диплом".lower()
    ),
    "q_entertainment": word_tokenize(
        "Какая внеучебная деятельность есть в МИСИС".lower()
    ),
    "q_enternship": word_tokenize(
        "Помогает ли МИСИС со стажировками для студентов?".lower()
    ),
    "q_budget": word_tokenize(
        "Есть ли в НИТУ МИСИС бюджетные места?".lower()
    ),
    "q_part_time": word_tokenize(
        "Есть ли очно-заочная либо заочная форма обучения?".lower()
    ),
    "q_dormitory": word_tokenize(
        "Предоставляется ли общежитие студентам НИТУ МИСИС?".lower()
    ),
    "q_dormitory_link": word_tokenize(
        "Где можно ознакомиться с информацией об общежитиях НИТУ МИСИС?".lower()
    ),
    "q_budget_and_pay": word_tokenize(
        "Можно ли подать документы на бюджетное и платное обучение одновременно?".lower()
    ),
    "q_transfer": word_tokenize(
        "Возможно ли перевестись с внебюджета на бюджет в процессе обучения?".lower()
    ),
    "q_foreign_budget": word_tokenize(
        "Могут ли иностранные граждане поступать на бюджетные места?".lower()
    ),
    "q_dnr_lnr": word_tokenize(
        "Я из ДНР/ЛНР. Каким образом я могу поступить в НИТУ МИСИС?".lower()
    )
}