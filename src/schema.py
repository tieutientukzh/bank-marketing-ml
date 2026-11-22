from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

Job = Literal[
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
    "self-employed", "services", "student", "technician", "unemployed", "unknown",
]
Marital = Literal["divorced", "married", "single", "unknown"]
Education = Literal[
    "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
    "professional.course", "university.degree", "unknown",
]
YesNoUnknown = Literal["no", "yes", "unknown"]
Contact = Literal["cellular", "telephone"]
Month = Literal["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DayOfWeek = Literal["mon", "tue", "wed", "thu", "fri"]
Poutcome = Literal["failure", "nonexistent", "success"]

CATEGORY_FIELDS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]

HIGH_RATE_CONTEXT = {
    "emp_var_rate": 1.1, "cons_price_idx": 93.994, "cons_conf_idx": -36.4,
    "euribor3m": 4.857, "nr_employed": 5191.0,
}
LOW_RATE_CONTEXT = {
    "emp_var_rate": -1.8, "cons_price_idx": 93.075, "cons_conf_idx": -47.1,
    "euribor3m": 1.405, "nr_employed": 5099.1,
}

SAMPLE_CUSTOMERS = {
    "high_chance": {
        "age": 22, "job": "student", "marital": "single", "education": "university.degree",
        "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
        "month": "mar", "day_of_week": "thu", "campaign": 1, "pdays": 3, "previous": 1,
        "poutcome": "success", **LOW_RATE_CONTEXT,
    },
    "typical": {
        "age": 40, "job": "admin.", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "day_of_week": "mon", "campaign": 2, "pdays": 999, "previous": 0,
        "poutcome": "nonexistent", **HIGH_RATE_CONTEXT,
    },
    "low_chance": {
        "age": 45, "job": "blue-collar", "marital": "married", "education": "basic.9y",
        "default": "unknown", "housing": "yes", "loan": "no", "contact": "telephone",
        "month": "may", "day_of_week": "fri", "campaign": 5, "pdays": 999, "previous": 0,
        "poutcome": "nonexistent", **HIGH_RATE_CONTEXT,
    },
}


class Customer(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [SAMPLE_CUSTOMERS["typical"]]})

    age: int = Field(ge=17, le=100)
    job: Job
    marital: Marital
    education: Education
    default: YesNoUnknown
    housing: YesNoUnknown
    loan: YesNoUnknown
    contact: Contact
    month: Month
    day_of_week: DayOfWeek
    campaign: int = Field(ge=1, le=60)
    pdays: int = Field(ge=0, le=999)
    previous: int = Field(ge=0, le=10)
    poutcome: Poutcome
    emp_var_rate: float = Field(ge=-5, le=3)
    cons_price_idx: float = Field(ge=90, le=97)
    cons_conf_idx: float = Field(ge=-60, le=-20)
    euribor3m: float = Field(ge=0, le=6)
    nr_employed: float = Field(ge=4500, le=5500)

    @model_validator(mode="after")
    def check_campaign_history(self):
        never_called = self.previous == 0
        if never_called != (self.poutcome == "nonexistent"):
            raise ValueError("poutcome phải là nonexistent khi và chỉ khi previous = 0")
        if never_called and self.pdays != 999:
            raise ValueError("previous = 0 thì pdays phải là 999 (chưa từng gọi)")
        if self.poutcome == "success" and self.pdays == 999:
            raise ValueError("poutcome = success thì pdays phải khác 999")
        return self


class Prediction(BaseModel):
    probability: float
    prediction: Literal["yes", "no"]
    threshold: float


def category_options():
    return {name: list(get_args(Customer.model_fields[name].annotation)) for name in CATEGORY_FIELDS}
