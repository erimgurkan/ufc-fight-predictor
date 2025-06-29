import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import shutil

# Clear screen
os.system('cls' if os.name == 'nt' else 'clear')
term_width = shutil.get_terminal_size().columns - 33
def indent(text): return " " * 33 + text

def center(text): return indent(text.center(term_width))
def input_centered(prompt_text):
    padding = (term_width - len(prompt_text)) // 2
    return input(" " * (padding + 28) + prompt_text)

def border():
    print(indent("=" * term_width))

def header_box(title):
    print(indent("╔" + "═" * (term_width - 2) + "╗"))
    print(indent("║" + title.center(term_width - 2) + "║"))
    print(indent("╚" + "═" * (term_width - 2) + "╝"))

def frame_start(): print(indent("╔" + "═" * (term_width - 2) + "╗"))
def frame_end(): print(indent("╚" + "═" * (term_width - 2) + "╝"))
def frame_line(content=""):
    print(indent("║" + content.center(term_width - 2) + "║"))

# Header
header_box("UFC FIGHT PREDICTOR")
frame_line("⚠️ Note: For accurate results, fighters should be in the same weight class (±1 max)")
frame_end()
print()
print(center("💥 Starting fight prediction using Machine Learning..."))

# Load data
fighters = pd.read_csv("fighter_stats.csv")
matches = pd.read_csv("large_dataset.csv")

# Aggregate stats
stats_columns = [
    'kd', 'sig_str', 'str', 'td', 'td_att', 'sub_att', 'rev', 'ctrl_sec',
    'wins_total', 'losses_total'
]

r_stats = matches[['r_fighter'] + [f'r_{col}' for col in stats_columns]]
b_stats = matches[['b_fighter'] + [f'b_{col}' for col in stats_columns]]
r_stats.columns = ['name'] + stats_columns
b_stats.columns = ['name'] + stats_columns

all_stats = pd.concat([r_stats, b_stats])
avg_stats = all_stats.groupby('name').mean().reset_index()

# Prime scores (manually defined)
prime_data = pd.DataFrame([
    {"name": "Ilia Topuria", "prime_score": 3},
    {"name": "Dustin Poirier", "prime_score": 0}
])

fighters = pd.merge(fighters, avg_stats, on='name', how='left')
fighters = pd.merge(fighters, prime_data, on='name', how='left')
fighters.fillna(0, inplace=True)

# Feature columns
features = [
    'reach_diff', 'height_diff', 'weight_diff',
    'SLpM_total_diff', 'SApM_total_diff', 'sig_str_acc_diff',
    'td_avg_diff', 'td_acc_diff', 'td_def_total_diff',
    'sub_avg_diff', 'str_def_total_diff',
    'wins_total_diff', 'losses_total_diff', 'kd_diff',
    'sig_str_diff', 'str_diff', 'td_diff', 'td_att_diff',
    'sub_att_diff', 'rev_diff', 'ctrl_sec_diff',
    'prime_score_diff'
]

def compute_diff(f1, f2):
    return pd.Series({
        'reach_diff': f1['reach'] - f2['reach'],
        'height_diff': f1['height'] - f2['height'],
        'weight_diff': f1['weight'] - f2['weight'],
        'SLpM_total_diff': f1['SLpM'] - f2['SLpM'],
        'SApM_total_diff': f1['SApM'] - f2['SApM'],
        'sig_str_acc_diff': f1['sig_str_acc'] - f2['sig_str_acc'],
        'td_avg_diff': f1['td_avg'] - f2['td_avg'],
        'td_acc_diff': f1['td_acc'] - f2['td_acc'],
        'td_def_total_diff': f1['td_def'] - f2['td_def'],
        'sub_avg_diff': f1['sub_avg'] - f2['sub_avg'],
        'str_def_total_diff': f1['str_def'] - f2['str_def'],
        'wins_total_diff': f1['wins_total'] - f2['wins_total'],
        'losses_total_diff': f1['losses_total'] - f2['losses_total'],
        'kd_diff': f1['kd'] - f2['kd'],
        'sig_str_diff': f1['sig_str'] - f2['sig_str'],
        'str_diff': f1['str'] - f2['str'],
        'td_diff': f1['td'] - f2['td'],
        'td_att_diff': f1['td_att'] - f2['td_att'],
        'sub_att_diff': f1['sub_att'] - f2['sub_att'],
        'rev_diff': f1['rev'] - f2['rev'],
        'ctrl_sec_diff': f1['ctrl_sec'] - f2['ctrl_sec'],
        'prime_score_diff': f1['prime_score'] - f2['prime_score']
    })

# Prepare training data
df = matches.dropna(subset=['r_fighter', 'b_fighter', 'winner'])
df_features = []

for _, row in df.iterrows():
    try:
        f1 = fighters[fighters['name'] == row['r_fighter']].iloc[0]
        f2 = fighters[fighters['name'] == row['b_fighter']].iloc[0]
        df_features.append(compute_diff(f1, f2))
    except:
        continue

df_features = pd.DataFrame(df_features)
df = df.iloc[:len(df_features)]
df[features] = df_features
df['winner_binary'] = df['winner'].apply(lambda x: 1 if x == 'Red' else 0)

X = df[features]
y = df['winner_binary']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(center(f"✅ Model trained successfully. Accuracy: {acc:.2f}"))

# Prediction function
def predict_fight(f1_name, f2_name):
    f1 = fighters[fighters['name'].str.lower() == f1_name.lower()]
    f2 = fighters[fighters['name'].str.lower() == f2_name.lower()]
    if f1.empty or f2.empty:
        return "❌ Fighter not found."

    input_df = pd.DataFrame([compute_diff(f1.iloc[0], f2.iloc[0])])
    proba = model.predict_proba(input_df)[0]
    return {
        "RedFighter": f1_name.title(),
        "BlueFighter": f2_name.title(),
        "RedWinProbability": round(proba[1] * 100, 2),
        "BlueWinProbability": round(proba[0] * 100, 2)
    }

# UI
print("\n" * 2)
print(center("👊 Type the fighters you want to simulate"))
red = input_centered("🔴 Red corner: ")
blue = input_centered("🔵 Blue corner: ")
print(center("🤖 Predicting outcome..."))

result = predict_fight(red, blue)
print()
frame_start()
if isinstance(result, dict):
    frame_line("🥊 PREDICTION RESULT")
    frame_line(f"🔴 Red Fighter : {result['RedFighter']}")
    frame_line(f"🔵 Blue Fighter: {result['BlueFighter']}")
    frame_line()
    frame_line(f"🏆 Red Win Probability : %{result['RedWinProbability']}")
    frame_line(f"💀 Blue Win Probability: %{result['BlueWinProbability']}")
else:
    frame_line(result)
frame_end()

input(center("⏎ Press Enter to exit..."))
