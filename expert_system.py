
import collections
import collections.abc
for _attr in ("Mapping", "MutableMapping", "Sequence"):
    if not hasattr(collections, _attr):
        setattr(collections, _attr, getattr(collections.abc, _attr))

import frozendict
import schema


from experta import KnowledgeEngine, Fact, Rule, Field, MATCH, DefFacts


class FeaturesFact(Fact):
    feature_name = Field(str, mandatory=True)
    shap_value = Field(float, mandatory=True)
    importance_level = Field(str, mandatory=True) 
    direction = Field(str, mandatory=True)         

class GenericExpertSystem(KnowledgeEngine):


    def __init__(self):
        super().__init__()
        self.explanations = []  

    @Rule(FeaturesFact(feature_name=MATCH.f,
                      shap_value=MATCH.v,
                      importance_level="High",
                      direction="Positive"))
    def strong_positive(self, f, v):
        self.explanations.append(
            f" الميزة '{f}' لها تأثير قوي إيجابي على القرار "
            f"(قيمة SHAP = {v:+.4f})، وتعد من أهم العوامل الداعمة للتنبؤ."
        )

    @Rule(FeaturesFact(feature_name=MATCH.f,
                      shap_value=MATCH.v,
                      importance_level="High",
                      direction="Negative"))
    def strong_negative(self, f, v):
        self.explanations.append(
            f" الميزة '{f}' لها تأثير قوي سلبي على القرار "
            f"(قيمة SHAP = {v:+.4f})، وتضعف ثقة النموذج في هذا التنبؤ."
        )

    @Rule(FeaturesFact(feature_name=MATCH.f,
                      shap_value=MATCH.v,
                      importance_level="Medium",
                      direction="Positive"))
    def medium_positive(self, f, v):
        self.explanations.append(
            f" الميزة '{f}' تساهم بشكل متوسط في تعزيز القرار "
            f"(قيمة SHAP = {v:+.4f})."
        )

    @Rule(FeaturesFact(feature_name=MATCH.f,
                      shap_value=MATCH.v,
                      importance_level="Medium",
                      direction="Negative"))
    def medium_negative(self, f, v):
        self.explanations.append(
            f" الميزة '{f}' تساهم بشكل متوسط في إضعاف القرار "
            f"(قيمة SHAP = {v:+.4f})."
        )

    @Rule(FeaturesFact(feature_name=MATCH.f,
                      shap_value=MATCH.v,
                      importance_level="Low"))
    def low_impact(self, f, v):
        self.explanations.append(
            f"الميزة '{f}' ذات تأثير ضعيف وغير مؤثر فعليا "
            f"(قيمة SHAP = {v:+.4f})."
        )


def classify_features(shap_dict, high_ratio=0.20, med_ratio=0.05):
    total = sum(abs(v) for v in shap_dict.values()) or 1e-9
    classified = []
    for name, val in shap_dict.items():
        ratio = abs(val) / total
        if ratio >= high_ratio:
            level = "High"
        elif ratio >= med_ratio:
            level = "Medium"
        else:
            level = "Low"
        direction = "Positive" if val >= 0 else "Negative"
        classified.append({
            "feature_name": name,
            "shap_value": float(val),
            "importance_level": level,
            "direction": direction,
            "ratio": ratio
        })
    return classified


def generate_arabic_report(shap_dict, predicted_class, class_names=None):
    classified = classify_features(shap_dict)
    engine = GenericExpertSystem()
    engine.reset()
    for item in sorted(classified, key=lambda x: -x["ratio"]):
        engine.declare(FeaturesFact(
            feature_name=item["feature_name"],
            shap_value=item["shap_value"],
            importance_level=item["importance_level"],
            direction=item["direction"]
        ))
    engine.run()
    label = (class_names[predicted_class]
             if class_names is not None else str(predicted_class))
    high_feats = [c for c in classified if c["importance_level"] == "High"]
    pos_high = [c for c in high_feats if c["direction"] == "Positive"]
    neg_high = [c for c in high_feats if c["direction"] == "Negative"]
    summary = (
        f" الملخص التنفيذي:\n"
        f"قام النموذج بالتنبؤ بالفئة: «{label}».\n"
        f"تم اكتشاف {len(high_feats)} ميزة ذات تأثير قوي، "
        f"{sum(1 for c in classified if c['importance_level']=='Medium')} ميزة متوسطة، "
        f"و {sum(1 for c in classified if c['importance_level']=='Low')} ميزة ضعيفة."
    )

    analysis_lines = ["تحليل الميزات المؤثرة:"]
    analysis_lines.extend(engine.explanations)

    if len(pos_high) > len(neg_high):
        conclusion = (
            f"الاستنتاج النهائي:\n"
            f"التنبؤ بـ «{label}» مدعوم بقوة من عدة ميزات إيجابية، "
            f"مما يجعل قرار النموذج موثوقا نسبيا."
        )
    elif len(neg_high) > len(pos_high):
        conclusion = (
            f"الاستنتاج النهائي:\n"
            f"رغم تنبؤ النموذج بـ «{label}»، توجد عوامل سلبية قوية "
        )
    else:
        conclusion = (
            f"الاستنتاج النهائي:\n"
            f"قرار النموذج بـ «{label}» متوازن بين العوامل المؤيدة والمعارضة، "
            f"ويحتاج إلى تدقيق إضافي."
        )

    report = "\n\n".join([summary, "\n".join(analysis_lines), conclusion])
    return report