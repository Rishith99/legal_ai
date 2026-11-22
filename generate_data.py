"""Creates the SYNTHETIC datasets used in this prototype:
   data/legal_cases.csv  - short case descriptions with a legal category
   data/advocates.csv    - fictional advocate profiles
Run: python generate_data.py"""
import os, random, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('data', exist_ok=True)
import pandas as pd

random.seed(42)

SLOTS = {
 "product": ["mobile phone", "washing machine", "laptop", "refrigerator", "air conditioner", "television", "sofa set", "smart watch", "bike", "microwave oven"],
 "shop": ["an online store", "a showroom", "the dealer", "an e-commerce company", "the retailer", "the company"],
 "service": ["coaching course", "gym membership", "travel package", "internet connection", "home repair service", "online subscription"],
 "c_issue": ["stopped working within a week", "was defective from day one", "was never delivered", "was a different model than advertised", "has a manufacturing fault", "got damaged in the first month"],
 "prop": ["flat", "house", "plot", "shop", "apartment", "farm land"],
 "delay": ["two years", "six months", "three years", "one year", "eight months", "five months"],
 "crime": ["theft", "cheating", "assault", "a fake dowry allegation", "forgery", "criminal intimidation", "a property dispute fight"],
 "rel": ["brother", "father", "husband", "friend", "son", "uncle"],
 "person": ["a neighbour", "my business partner", "an unknown person", "my relative", "a local gang", "my colleague"],
 "item": ["mobile phone", "bike", "gold chain", "wallet", "laptop", "car"],
 "account": ["Instagram", "Facebook", "email", "WhatsApp", "bank", "Gmail"],
 "vehicle": ["car", "bike", "scooter", "auto rickshaw", "truck"],
 "amt": ["15,000", "50,000", "2 lakh", "75,000", "5 lakh", "1.5 lakh", "30,000", "10 lakh"],
 "time": ["month", "week", "year", "Diwali", "monsoon"],
}

CATS = {
 "Consumer Protection": [
  "I bought a {product} from {shop} for Rs {amt} and it {c_issue}. The seller is refusing to give a refund.",
  "The {product} I ordered online {c_issue}. Customer care is not responding to my complaints.",
  "{shop} charged me more than the MRP for a {product} and did not give a proper bill.",
  "My warranty claim for the {product} was rejected even though it {c_issue}.",
  "I paid Rs {amt} for a {service} but the company did not provide it and refuses to return my money.",
  "The {product} delivered by {shop} was damaged and they refuse replacement or compensation.",
  "I want to file a consumer complaint because the {service} was misleading and the company will not refund Rs {amt}.",
 ],
 "Property & Tenancy": [
  "My landlord is not returning my security deposit of Rs {amt} after I vacated the {prop}.",
  "The builder has not handed over my {prop} even after {delay} of paying the full amount.",
  "A neighbour has encroached on my {prop} and built a wall on my land.",
  "My landlord is trying to evict me from the rented {prop} without any notice.",
  "There is a dispute with my brother over the ancestral {prop} and the sale deed is not registered.",
  "I want to verify the title documents before buying a {prop} because the seller is hiding some papers.",
  "The tenant in my {prop} has not paid rent for {delay} and refuses to vacate.",
 ],
 "Family & Matrimonial": [
  "My husband and I have been separated for {delay} and I want to file for divorce.",
  "My wife left home with our child and I want custody of the child.",
  "My in-laws are harassing me for dowry and I need maintenance for myself and my child.",
  "We both want a mutual consent divorce and need to know the procedure.",
  "My father passed away without a will and my siblings are denying my share in the family property.",
  "I need alimony and maintenance from my spouse after the separation of {delay}.",
  "I want to legally adopt a child and need guidance on the adoption process.",
 ],
 "Criminal": [
  "Police registered a false FIR against me for {crime}. I need anticipatory bail.",
  "My {rel} was arrested for {crime} and we need a bail lawyer urgently.",
  "I was threatened and assaulted by {person} and the police are not registering my complaint.",
  "Someone stole my {item} and I want to file an FIR but the police refuse.",
  "I have received a court summons in a criminal case about {crime}.",
  "I need a criminal defence lawyer because I am falsely accused of {crime}.",
 ],
 "Labour & Employment": [
  "My employer terminated me without notice and has not paid my salary for {delay}.",
  "The company is not paying my pending salary, PF and gratuity after I resigned.",
  "I faced harassment at the workplace and HR is ignoring my written complaint.",
  "My employer forced me to sign a resignation letter and refuses to give an experience certificate.",
  "I was denied maternity leave benefits by my company and then removed from the job.",
  "My company withheld my full and final settlement of Rs {amt} after I left the job.",
 ],
 "Cyber Crime": [
  "Someone hacked my {account} account and is demanding Rs {amt} to return it.",
  "I lost Rs {amt} in an online UPI fraud after clicking a fake link.",
  "{person} is posting my morphed photos on social media and blackmailing me.",
  "I received a fake call asking for my OTP and money was debited from my account.",
  "Someone is cyberstalking me with threatening messages on {account}.",
  "A fake profile of mine on {account} is being used to cheat my friends online.",
 ],
 "Motor Accident": [
  "My {vehicle} met with an accident caused by a rash truck driver and I need compensation.",
  "My {rel} died in a road accident and the insurance company is delaying the claim.",
  "A hit and run accident injured me badly and the driver could not be traced.",
  "The insurance company rejected my {vehicle} damage claim after the accident.",
  "I was injured when a bus hit my {vehicle} and I want a motor accident claim for hospital expenses of Rs {amt}.",
 ],
 "Banking & Cheque Bounce": [
  "A cheque of Rs {amt} given to me by {person} bounced due to insufficient funds.",
  "The bank is harassing me for loan recovery and sent a recovery notice.",
  "I received a legal notice under Section 138 for a bounced cheque.",
  "The bank wrongly deducted charges and refuses to correct my loan statement.",
  "A loan recovery agent is threatening me for a missed EMI payment of Rs {amt}.",
  "My business partner issued a cheque of Rs {amt} which got dishonoured and he is not paying.",
 ],
}
GENERIC = ["This happened last {time}.", "I want to know my legal options.", "Please guide me on what to do next.",
           "I have all the documents with me.", "I am very stressed about this.", "It has been going on for a long time.",
           "I paid Rs {amt} in total.", "Please suggest a good lawyer."]

def fill(t):
    return re.sub(r"\{(\w+)\}", lambda m: random.choice(SLOTS[m.group(1)]), t)

def make_case(cat):
    parts = [fill(random.choice(CATS[cat]))]
    if random.random() < 0.5:
        parts.append(fill(random.choice(GENERIC)))
    if random.random() < 0.20:   # confusing sentence borrowed from another category
        other = random.choice([c for c in CATS if c != cat])
        parts.insert(random.randint(0, 1), fill(random.choice(CATS[other])))
    text = " ".join(parts)
    if random.random() < 0.25:
        text = text.lower()
    return text

rows = []
for cat in CATS:
    for _ in range(300):
        rows.append((make_case(cat), cat))
df = pd.DataFrame(rows, columns=["description", "category"]).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/legal_cases.csv", index=False)

# ------------------------------------------------------------ advocates
first = ["Rahul", "Priya", "Arjun", "Sneha", "Vikram", "Anita", "Karthik", "Meera", "Rohan", "Divya", "Suresh", "Kavya",
         "Imran", "Neha", "Aditya", "Pooja", "Sanjay", "Lakshmi", "Faisal", "Ritu"]
last = ["Sharma", "Reddy", "Iyer", "Khan", "Patel", "Nair", "Gowda", "Singh", "Menon", "Das", "Rao", "Verma"]
cities = ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata", "Mysuru"]
langs = ["English", "Hindi", "Kannada", "Tamil", "Telugu", "Marathi", "Bengali", "Urdu"]
cat_list = list(CATS)
adv = []
for i in range(1, 61):
    name = f"Adv. {random.choice(first)} {random.choice(last)}"
    p = cat_list[(i - 1) % len(cat_list)]
    s = random.choice([c for c in cat_list if c != p]) if random.random() < 0.5 else ""
    n_lang = random.choice([2, 2, 3])
    lg = ["English"] + random.sample(langs[1:], n_lang - 1)
    adv.append(dict(advocate_id=f"ADV{i:03d}", name=name, primary_specialization=p, secondary_specialization=s,
                    experience_years=random.randint(2, 28), rating=round(random.uniform(3.4, 5.0), 1),
                    city=random.choice(cities), languages=";".join(lg), cases_handled=random.randint(30, 600)))
pd.DataFrame(adv).to_csv("data/advocates.csv", index=False)
print(f"Created {len(df)} case descriptions and {len(adv)} advocates")
print(df["category"].value_counts().to_string())
