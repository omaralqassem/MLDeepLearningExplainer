import collections
import collections.abc
for _attr in ("Mapping", "MutableMapping", "Sequence"):
    if not hasattr(collections, _attr):
        setattr(collections, _attr, getattr(collections.abc, _attr))

from experta import KnowledgeEngine, Fact, Rule, Field, MATCH
import shap
import numpy as np
import pandas as pd

MEDICAL_METADATA = {
    "systolic_bp": {
        "unit": "mmHg",
        "group": "Vitals",
        "arabic_name": "ضغط الدم الانقباضي",
        "ranges": {
            "default": "90-120",
            "elderly": "90-130",
            "diabetic": "90-130"  
        }
    },
    "ldl_cholesterol": {
        "unit": "mg/dL",
        "group": "Labs",
        "arabic_name": "الكوليسترول الضار (LDL)",
        "ranges": {
            "default": "<100",
            "diabetic": "<70"     
        }
    },
    "creatinine": {
        "unit": "mg/dL",
        "group": "Labs",
        "arabic_name": "الكرياتينين في الدم",
        "ranges": {
            "male": "0.7-1.3",
            "female": "0.6-1.1",
            "default": "0.6-1.2"
        }
    },
    "urea": {
        "unit": "mg/dL",
        "group": "Labs",
        "arabic_name": "اليوريا في الدم",
        "ranges": {
            "default": "7-20"
        }
    },
    "glucose": {
        "unit": "mg/dL",
        "group": "Labs",
        "arabic_name": "جلوكوز الدم الصائم",
        "ranges": {
            "default": "70-100",
            "diabetic": "80-130" 
            }
    },
    "heart_rate": {
        "unit": "bpm",
        "group": "Vitals",
        "arabic_name": "معدل ضربات القلب",
        "ranges": {
            "default": "60-100"
        }
    },
}


class PatientContext(Fact):
    age = Field(int, mandatory=True)
    gender = Field(str, mandatory=True) 
    comorbidities = Field(list, default=[])


class FeaturesFact(Fact):
    feature_name = Field(str, mandatory=True)
    raw_name = Field(str, mandatory=True)
    value = Field(float, mandatory=True)
    unit = Field(str, default="")
    normal_range = Field(str, default="")
    group = Field(str, default="General")
    shap_value = Field(float, mandatory=True)
    importance_level = Field(str, mandatory=True)
    direction = Field(str, mandatory=True)


class MedicalExpertSystem(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.explanations = []
        self.clinical_insights = []

    # قاعدة: عوامل الخطورة عالية التأثير
    @Rule(FeaturesFact(feature_name=MATCH.f,
                       value=MATCH.val,
                       unit=MATCH.u,
                       normal_range=MATCH.nr,
                       shap_value=MATCH.v,
                       importance_level="High",
                       direction="Positive"))
    def rule_high_risk_factor(self, f, val, u, nr, v):
        ref = f" [المعدل الطبيعي المرجعي المخصص: {nr} {u}]" if nr else ""
        self.explanations.append(
            f"• يمثل مؤشر '{f}' بقيمته المسجلة ({val} {u}){ref} عامل خطورة مرتفع الأثر، "
            f"مما يرجح هذا التنبؤ الطبي سريرياً بنسبة تأثير واضحة."
        )

    # قاعدة: المؤشرات المطمئنة عالية التأثير 
    @Rule(FeaturesFact(feature_name=MATCH.f,
                       value=MATCH.val,
                       unit=MATCH.u,
                       normal_range=MATCH.nr,
                       shap_value=MATCH.v,
                       importance_level="High",
                       direction="Negative"))
    def rule_high_protective_factor(self, f, val, u, nr, v):
        ref = f" [المعدل الطبيعي المرجعي المخصص: {nr} {u}]" if nr else ""
        self.explanations.append(
            f"• يمثل مؤشر '{f}' بقيمته المسجلة ({val} {u}){ref} مؤشراً مطمئناً قوي الأثر (عامل حماية)، "
            f"ويسهم بوضوح في خفض احتمالية الإصابة أو استبعاد التنبؤ الحالي."
        )

    @Rule(FeaturesFact(feature_name=MATCH.f,
                       value=MATCH.val,
                       unit=MATCH.u,
                       normal_range=MATCH.nr,
                       shap_value=MATCH.v,
                       importance_level="Medium",
                       direction="Positive"))
    def rule_medium_risk_factor(self, f, val, u, nr, v):
        ref = f" [المعدل الطبيعي المرجعي المخصص: {nr} {u}]" if nr else ""
        self.explanations.append(
            f"• يظهر مؤشر '{f}' بقيمته المسجلة ({val} {u}){ref} كعامل خطورة متوسط ومساهم في القرار الحالي."
        )

    # قاعدة: المؤشرات المطمئنة متوسطة التأثير
    @Rule(FeaturesFact(feature_name=MATCH.f,
                       value=MATCH.val,
                       unit=MATCH.u,
                       normal_range=MATCH.nr,
                       shap_value=MATCH.v,
                       importance_level="Medium",
                       direction="Negative"))
    def rule_medium_protective_factor(self, f, val, u, nr, v):
        ref = f" [المعدل الطبيعي المرجعي المخصص: {nr} {u}]" if nr else ""
        self.explanations.append(
            f"• يمثل مؤشر '{f}' بقيمته المسجلة ({val} {u}){ref} مؤشراً مطمئناً متوسط المساهمة."
        )

    # قاعدة: المؤشرات منخفضة التأثير
    @Rule(FeaturesFact(feature_name=MATCH.f,
                       value=MATCH.val,
                       unit=MATCH.u,
                       shap_value=MATCH.v,
                       importance_level="Low"))
    def rule_low_impact_factor(self, f, val, u, v):
        self.explanations.append(
            f"• مؤشر '{f}' بقيمته الحالية ({val} {u}) ذو تأثير محدود وغير حاسم في مسار القرار السريري الحالي."
        )

    # قاعدة تفاعل سريري وعائي: ارتفاع ضغط الدم والدهون معاً
    @Rule(
        FeaturesFact(group="Vitals", raw_name=MATCH.rn1, direction="Positive", importance_level="High"),
        FeaturesFact(group="Labs", raw_name=MATCH.rn2, direction="Positive", importance_level="High")
    )
    def check_cardio_synergy(self, rn1, rn2):
        bp_list = ["systolic_bp", "blood_pressure"]
        chol_list = ["ldl_cholesterol", "cholesterol"]
        is_bp = rn1 in bp_list or rn2 in bp_list
        is_chol = rn1 in chol_list or rn2 in chol_list
        if is_bp and is_chol:
            self.clinical_insights.append(
                " [تحذير سريري]: لوحظ تأثير تآزري سلبي ناتج عن التزامن بين ارتفاع ضغط الدم الشرياني وارتفاع الكوليسترول الضار، "
                "مما يزيد بشكل مضاعف من مخاطر الحوادث القلبية والوعائية مقارنة بوجود عامل منفرد."
            )

    # قاعدة تفاعل سريري كلوي: ارتفاع الكرياتينين واليوريا معاً
    @Rule(
        FeaturesFact(group="Labs", raw_name=MATCH.rn1, direction="Positive", importance_level="High"),
        FeaturesFact(group="Labs", raw_name=MATCH.rn2, direction="Positive", importance_level="High")
    )
    def check_renal_synergy(self, rn1, rn2):
        if rn1 != rn2:
            is_creat = rn1 == "creatinine" or rn2 == "creatinine"
            is_urea = rn1 == "urea" or rn2 == "urea"
            if is_creat and is_urea:
                self.clinical_insights.append(
                    " [تحذير سريري]: يظهر الارتفاع المشترك والبارز في الكرياتينين واليوريا كدلالة قوية "
                    "على احتمالية وجود تراجع ملحوظ في معدل الترشيح الكبيبي والوظيفة الكلوية للحالة."
                )

    # قاعدة دمج سياق المريض: مريض السكري المصاب بارتفاع ضغط الدم
    @Rule(
        PatientContext(comorbidities=MATCH.comor),
        FeaturesFact(raw_name="systolic_bp", value=MATCH.val, direction="Positive", importance_level="High")
    )
    def check_diabetic_hypertension_context(self, comor, val):
        if "diabetes" in comor:
            self.clinical_insights.append(
                f" [تنبيه خاص بالسكري]: إن رصد ارتفاع ضغط الدم الانقباضي بمستوى ({val} mmHg) "
                f"يمثل تهديداً متزايداً للإصابة باعتلال الشبكية والكلية لدى مرضى السكري بشكل خاص، "
                f"ويتطلب خطة علاجية مستهدفة لضبط الضغط في مستويات أدنى."
            )

    # قاعدة دمج سياق المريض: المريض المتقدم في السن
    @Rule(
        PatientContext(age=MATCH.age),
        FeaturesFact(raw_name="ldl_cholesterol", value=MATCH.val, direction="Positive", importance_level="High")
    )
    def check_elderly_cholesterol_context(self, age, val):
        if age > 65:
            self.clinical_insights.append(
                f" [تنبيه سريري لفئة كبار السن]: المريض يبلغ من العمر ({age} عاماً) ولديه ارتفاع "
                f"في مؤشر LDL بمقدار ({val} mg/dL). يتطلب التدخل الدوائي عناية خاصة لضمان موازنة الفوائد "
                f"مقابل الآثار الجانبية للأدوية الخافضة للدهون."
            )


def classify_features_clinical(shap_dict, sample_dict, patient_context, high_ratio=0.20, med_ratio=0.05):
    total = sum(abs(v) for v in shap_dict.values()) or 1e-9
    classified = []
    
    age = patient_context.get("age", 40)
    gender = patient_context.get("gender", "male")
    comorbidities = patient_context.get("comorbidities", [])

    for name, val in shap_dict.items():
        ratio = abs(val) / total
        if ratio >= high_ratio:
            level = "High"
        elif ratio >= med_ratio:
            level = "Medium"
        else:
            level = "Low"
            
        direction = "Positive" if val >= 0 else "Negative"
        
        meta = MEDICAL_METADATA.get(name.lower(), {})
        arabic_name = meta.get("arabic_name", name)
        unit = meta.get("unit", "")
        group = meta.get("group", "General")
        ranges = meta.get("ranges", {})
        if "diabetes" in comorbidities and "diabetic" in ranges:
            normal_range = ranges["diabetic"]
        elif age > 65 and "elderly" in ranges:
            normal_range = ranges["elderly"]
        elif gender in ranges:
            normal_range = ranges[gender]
        else:
            normal_range = ranges.get("default", "")
        
        raw_val = float(sample_dict.get(name, 0.0))
        
        classified.append({
            "feature_name": arabic_name,
            "raw_name": name.lower(),
            "value": raw_val,
            "unit": unit,
            "normal_range": normal_range,
            "group": group,
            "shap_value": float(val),
            "importance_level": level,
            "direction": direction,
            "ratio": ratio
        })
        
    return classified


def generate_arabic_medical_report(shap_dict, sample_dict, predicted_class, patient_context, class_names=None):
    classified = classify_features_clinical(shap_dict, sample_dict, patient_context)
    
    engine = MedicalExpertSystem()
    engine.reset()    
    engine.declare(PatientContext(
        age=patient_context.get("age", 40),
        gender=patient_context.get("gender", "male"),
        comorbidities=patient_context.get("comorbidities", [])
    ))    
    for item in sorted(classified, key=lambda x: -x["ratio"]):
        engine.declare(FeaturesFact(
            feature_name=item["feature_name"],
            raw_name=item["raw_name"],
            value=item["value"],
            unit=item["unit"],
            normal_range=item["normal_range"],
            group=item["group"],
            shap_value=item["shap_value"],
            importance_level=item["importance_level"],
            direction=item["direction"]
        ))
        
    engine.run()
    
    label = (class_names[predicted_class] if class_names is not None else str(predicted_class))
    
    high_feats = [c for c in classified if c["importance_level"] == "High"]
    risk_high = [c for c in high_feats if c["direction"] == "Positive"]
    prot_high = [c for c in high_feats if c["direction"] == "Negative"]
    
    summary_section = (
        f"التقرير التشخيصي الاستشاري المساعد لبيانات المريض\n"
        f"============================================\n"
        f" التشخيص المرجح من النموذج: « {label} »\n"
        f" العمر: {patient_context.get('age')} | الجنس: {patient_context.get('gender')} | الأمراض المصاحبة: {', '.join(patient_context.get('comorbidities', [])) or 'لا يوجد'}\n"
        f" فحص المؤشرات المؤثرة:\n"
        f"  - عوامل خطورة عالية الأثر: {len(risk_high)}\n"
        f"  - مؤشرات مطمئنة عالية التأثير: {len(prot_high)}\n"
        f"  - مؤشرات متوسطة التأثير: {sum(1 for c in classified if c['importance_level']=='Medium')}\n"
        f"  - مؤشرات طفيفة الأثر: {sum(1 for c in classified if c['importance_level']=='Low')}"
    )
    analysis_section = ["\n تحليل المعطيات الحيوية والمخبرية للمريض:"]
    analysis_section.extend(engine.explanations)
    synergy_section = []
    if engine.clinical_insights:
        synergy_section.append("\n تنبيهات سريرية ناتجة عن سياق المريض وتفاعل المؤشرات:")
        synergy_section.extend(engine.clinical_insights)

    if len(risk_high) > len(prot_high):
        conclusion_section = (
            f"\n التقييم النهائي والتوصية:\n"
            f"القرار التشخيصي الحالي يستند إلى عوامل خطورة متعددة وواضحة الأثر. "
            f"يُرجى مطابقة هذه الدلائل مخبرياً وسريرياً مع الحالة السريرية العامة للمريض."
        )
    elif len(prot_high) > len(risk_high):
        conclusion_section = (
            f"\n التقييم النهائي والتوصية:\n"
            f"على الرغم من تشخيص النموذج المقترح، تتوفر مؤشرات مطمئنة قوية الأثر "
            f"قد تسهم في استقرار الحالة. يوصى بالمتابعة ومراعاة التباينات الفردية للمريض."
        )
    else:
        conclusion_section = (
            f"\n التقييم النهائي والتوصية:\n"
            f"تبين وجود توازن متقارب وتداخل مستمر بين عوامل الخطورة والمؤشرات المطمئنة المكتشفة. "
            f"الحالة تستوجب فحصاً وتدقيقاً طبياً مباشراً لتقييم المخاطر بشكل أدق."
        )

    sections = [summary_section, "\n".join(analysis_section)]
    if synergy_section:
        sections.append("\n".join(synergy_section))
    sections.append(conclusion_section)
    
    return "\n\n".join(sections)

