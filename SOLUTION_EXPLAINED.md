# MITRA — The Solution, Explained Simply

*MITRA (Model-driven Insights for Traffic & Routing Assistance) — "mitra" means friend.
A plain-language walkthrough of what I built, how it answers each part of the problem,
what I tried before, and how accurate it is. No ML background needed.*

---

## 1. The problem in one paragraph

Events — political rallies, festivals, construction, breakdowns, accidents, VIP
movements — cause sudden, localized traffic jams in Bengaluru. Today the traffic police
react *after* a jam forms, decide how many officers and barricades to send based on
gut/experience, and don't have a system that learns from past events. The challenge:
**use historical data to (a) predict an event's traffic impact in advance, and
(b) recommend how much manpower, barricading, and diversion to deploy.**

I was given an anonymised **Bengaluru Traffic Police** dataset — a log of **8,057 real
traffic incidents** reported between November 2023 and April 2024. Each row is one event
with its location, time, cause, the road/corridor, whether it needed a road closure, and
(sometimes) how long it took to clear.

---

## 2. How I interpreted each part of the problem

The data doesn't contain a column literally called "traffic impact." So I had to decide
which real, recorded facts best *stand in* for the things the problem asks about. This is
the most important design step, so here it is spelled out:

| Problem statement asks for… | What I used as the measurable stand-in | Why |
|---|---|---|
| "Event **impact**" | **Whether the event required a road closure** (a recorded yes/no) | A closure is the clearest sign an event seriously disrupted traffic. |
| "Impact" (severity & duration) | **How long the event took to clear** (minutes) | Longer clearance = more disruption and more manpower-hours needed. |
| "Forecast **event-related** congestion" | **How many events will hit each road, each hour** | If you know where/when events cluster, you can pre-position resources. |
| "Recommend **manpower / barricading / diversion**" | **Rules that convert the predictions into a deployment plan** | The predictions are only useful if they become concrete actions. |
| "No **post-event learning** today" | **The model is trained on history and can be re-trained** | Every new event becomes training data; the system improves over time. |

A note I was careful about: the data has a `priority` (High/Low) column, but I
found it is just an **administrative label** — it's "High" essentially whenever the
event is on a major named corridor (99.9% of the time). So predicting it is meaningless
(you'd just be re-stating a clerical rule), and I do **not** count it as a real result.
I flag this honestly instead of inflating my accuracy with it.

---

## 3. The solution at a glance

Think of it as **four cooperating parts**:

```
  An event is reported  ─►  [1] Will it need a road closure?   (probability 0–100%)
                            [2] How long until it clears?       (minutes)
                                        │
  Looking ahead in time ─►  [3] Which roads will be busy with events, and when?
                                        │
                                        ▼
                            [4] Deployment plan:  how many officers,
                                how many barricades, what diversion
```

Parts 1–2 act the moment an event is reported. Part 3 looks into the future to help
*pre-position* resources before events even happen. Part 4 turns all of it into orders a
field officer can act on.

---

## 4. Each part explained — and how accurate it is

### What kind of model is this, in plain terms?
The core engine is a **"gradient-boosted decision tree."** Picture a flowchart of simple
yes/no questions ("Is the cause a VIP movement? Is it on Bellary Road? Is it near the city
centre?"). One flowchart is weak, so the model builds **hundreds of small flowcharts, each
fixing the mistakes of the last**, and averages them. It's the standard, battle-tested
approach for this kind of spreadsheet-like data — consistently stronger than fancier
methods here, as my experiments confirmed.

---

### [1] Will the event need a road closure? → drives **barricading & diversion**

The model reads the event's cause, road, vehicle type, location and the words in the
report, and outputs a **probability of needing a road closure**.

- **Accuracy: AUC = 0.81.** AUC is "given one event that needed a closure and one that
  didn't, how often does the model rank the real one as more likely?" 0.5 is coin-flip,
  1.0 is perfect — **0.81 is solidly good**, especially because closures are rare (only
  ~7% of events), which is a hard needle-in-haystack setting.
- Its probabilities are **well-calibrated** — when it says "30% chance," about 30% of such
  events really do close — which matters because the deployment plan uses these numbers
  directly.
- **What it learned (sanity check):** the biggest driver is the **cause** (VIP movement,
  public events, tree-fall and construction close roads far more than breakdowns), then
  the **road**, then **distance from the city centre**. This matches real-world intuition.

### [2] How long until it clears? → drives **manpower (how long to deploy)**

Two models answer this:
- A **clearance-time estimator**: average error **±95 minutes**, but the *typical* (median)
  error is only **±32 minutes** — the big average is dragged up by a few rare marathon
  incidents. For comparison, a naive "just guess the overall average every time" approach
  is off by ~108 minutes, so the model is clearly better.
- A **survival model** (the kind used in medicine for "how long until an event happens").
  Its advantage: ~two-thirds of incidents in the data **never had an end-time logged**.
  A plain estimator throws those away; the survival model **still uses them** (it knows
  they lasted "at least X minutes"). It **ranks clearance times better** than the plain
  estimator (a score of **0.715 vs 0.692**), using **706 extra events** that would
  otherwise be wasted.

### [3] Which roads will be busy, and when? → drives **pre-positioning**

This part forecasts the **number of events expected on each corridor in each upcoming
hour**. It learns the daily/weekly rhythm (rush hours, weekends) plus **spillover from
neighbouring roads**.

- **Accuracy: +10.4% better than a strong "same hour last week" baseline**, and it stays
  consistent when tested repeatedly across different time windows (error 0.146 ± 0.018
  events per road-hour). In short: it reliably highlights tomorrow's likely hotspots.

### [4] The deployment plan → the actual recommendation

A transparent set of rules turns the three predictions into orders, e.g.:

> *Public event on Hosur Road → 29% closure risk, "High" severity →*
> **4 officers, 4 barricades, advisory diversion (warn drivers, suggest alternate route).**

**How the severity tier is decided.** Each event gets a 0–1 severity score that blends the
three signals that actually reflect impact:

```
severity = 0.55 × closure-probability   (the acute disruption: needs barricades/diversion)
         + 0.35 × expected-clearance-time (resource-hours the event ties up)
         + 0.10 × disruptive-cause flag   (gatherings, VIP, construction, accidents …)
```
→ **Low** < 0.10 · **Moderate** < 0.20 · **High** < 0.46 · **Critical** ≥ 0.46, with a guard
that a near-certain closure is escalated on its own.

I didn't *guess* those weights — I **tuned them** so the score best *ranks events by what
actually happened* (real recorded road closures + clearance times on held-out data; see
`tune_severity.py`). An earlier version mistakenly gave 30% weight to an administrative
"is this on a major corridor?" flag, which pushed almost every event to "High". After the
fix the tiers separate cleanly — the real closure rate rises **1% → 7% → 19% → 35%** from
Low to Critical — and a routine breakdown on a busy corridor now correctly reads **Low**,
not High.

I deliberately kept this part **rule-based and readable** (not a black box) so officers
can understand and override it — important for real-world trust.

---

## 5. What I tried before, and why the current version is better

I didn't arrive here in one shot. The improvements came from **testing honestly**, not
guessing. Key steps:

| What I tried | What happened | What I did |
|---|---|---|
| Threw in **every feature I could engineer** (40+: density maps, recency, junction/police-station codes, etc.) | More features actually made the closure model **worse** (it overfit on noise) | **Removed** the unhelpful ones — a *leaner* model scored higher |
| Assumed the "priority" column was a real target | Found it's just a **clerical rule** (= "is it a main road") — fake 100% accuracy | **Dropped it** as a headline result; reported the honesty |
| Added spatial "heat-map" density features everywhere | They **helped clearance-time** but **hurt closure prediction** | Used them **only where they help** (duration), not everywhere |
| Used one simple train/test split | One split can be a lucky/unlucky fluke | Switched to **repeated time-based testing** (train on the past, test on the future, several times) — the honest way to judge a forecaster |
| Ignored events with no logged end-time | Wasted ~two-thirds of the duration data | Added the **survival model** to use those censored events |
| Only one model type | Wondered if a fancier model would win | **Compared 5 model families** — they tied, so I kept the simplest, well-calibrated one |

**Concrete result of all this:** the road-closure model improved from **AUC 0.787 → 0.812**,
the clearance-time error dropped, and — just as important — I now *know* which pieces
actually contribute, instead of hoping they do.

I also checked **what the dataset's own creators and the research literature recommend**.
That's where the **survival model** came from. (I deliberately did **not** use heavy
"deep learning on road networks," because that needs live road-speed sensor data I don't
have — using it would be cargo-culting, not engineering.)

---

## 6. Accuracy summary (one table)

| What it predicts | Metric (plain meaning) | Score | Reference point |
|---|---|---|---|
| Needs a road closure? | AUC (ranking quality, 0.5–1.0) | **0.81** | 0.5 = guessing |
| Needs a road closure? | Probabilities are trustworthy? | **Well-calibrated** | — |
| Clearance time | Typical error | **±32 min** (median) | naive guess ≈ ±108 min |
| Clearance time | Ranking quality (survival) | **0.715** | beats plain model's 0.692 |
| Event hotspots | vs. "same hour last week" | **+10.4% better** | 0% = no improvement |

All scores are measured on **data the model never saw during training**, using
**future** time periods — so they reflect real forecasting ability, not memorization.

---

## 7. Honest limitations (so expectations are right)

- The data logs **incidents**, not actual traffic speed. I predict *event impact signals*
  (closures, clearance time, event counts), which is the best the data allows — not a live
  speed/jam map.
- Clearance times were only recorded for ~1 in 3 events; my numbers are good **planning
  estimates**, not guarantees.
- Road-closures are genuinely rare (7%), so the closure model is best used to **rank and
  prioritise** events rather than as a hard yes/no switch.
- With more data (more months, live feeds) every part of this would sharpen — the system
  is built to **keep learning** as new events arrive, which directly answers the "no
  post-event learning today" gap in the problem statement.
