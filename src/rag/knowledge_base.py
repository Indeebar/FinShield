"""
Fraud Case Knowledge Base — FinShield
50 synthetic fraud case documents used by the RAG explainability engine.
Each case has a pattern, context, and outcome description.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class FraudCase:
    case_id:     str
    pattern:     str          # short pattern tag (used for metadata filtering)
    title:       str          # one-line summary
    description: str          # full natural language description for embedding
    amount_range: str         # "low", "medium", "high", "very_high"
    time_context: str         # "night", "day", "weekend", "any"
    outcome:     str          # what happened / how it was caught


FRAUD_CASES: List[FraudCase] = [

    # --- Card Not Present (CNP) ---
    FraudCase("CNP-001", "card_not_present",
        "High-value online purchase at night — card stolen",
        "Fraudster used stolen card details for a large online purchase at 2AM. "
        "Amount was 9x above the cardholder's 30-day average. No 3DS verification triggered. "
        "Merchant was a new electronics retailer. V14 showed strong deviation.",
        "high", "night",
        "Card blocked after second transaction. Cardholder confirmed card stolen from wallet."),

    FraudCase("CNP-002", "card_not_present",
        "Rapid multiple small purchases — card testing",
        "Fraudster tested stolen card with 5 purchases under $5 within 10 minutes across "
        "different merchants. Classic card testing pattern before large purchase. "
        "High rolling_count_1h, very low individual amounts.",
        "low", "any",
        "Velocity threshold triggered. Account locked after 5 micro-transactions."),

    FraudCase("CNP-003", "card_not_present",
        "International digital purchase — geo-anomaly",
        "Transaction originated from different country than cardholder's home country. "
        "Single high-value digital goods purchase. V3, V10 showed high deviation. "
        "Transaction at unusual hour for home timezone.",
        "very_high", "night",
        "Flagged by geo-velocity check. Cardholder was home, confirmed fraud."),

    FraudCase("CNP-004", "card_not_present",
        "Subscription service sign-up — phishing",
        "Series of subscription service sign-ups after phishing attack. "
        "Small monthly amounts, many merchants in 24h. High rolling_count_24h. "
        "V7 and V20 showed unusual patterns.",
        "low", "day",
        "Pattern matched known phishing campaign. 12 accounts affected."),

    FraudCase("CNP-005", "card_not_present",
        "Single large luxury purchase — data breach use",
        "Card data obtained from data breach. Used immediately for high-value luxury purchase. "
        "Amount 15x above normal. First transaction on card in 3 months. "
        "V12 and V17 heavily deviated from baseline.",
        "very_high", "day",
        "Purchase declined after real-time scoring. Cardholder alerted."),

    # --- Account Takeover (ATO) ---
    FraudCase("ATO-001", "account_takeover",
        "Password reset followed by large transfer — ATO",
        "Account credentials compromised via credential stuffing. "
        "Password changed, then immediate high-value transfer initiated. "
        "V1 and V5 showed extreme deviation. Transaction 20x above normal.",
        "very_high", "night",
        "Account takeover confirmed. Transfer reversed within 2 hours."),

    FraudCase("ATO-002", "account_takeover",
        "New payee added then transfer — social engineering",
        "Fraudster called bank posing as fraud team. Convinced customer to add new payee. "
        "Immediate large transfer to mule account. amount_vs_mean_ratio > 25.",
        "very_high", "day",
        "Transfer flagged. Customer called back to verify. Fraud confirmed."),

    FraudCase("ATO-003", "account_takeover",
        "Multiple failed logins then transaction — brute force",
        "15 failed login attempts followed by successful login from new device. "
        "Immediate ATM withdrawal at max daily limit. "
        "is_night = True, rolling_count_1h spike.",
        "high", "night",
        "Brute force pattern detected. Account locked. Customer notified."),

    FraudCase("ATO-004", "account_takeover",
        "SIM swap — OTP bypassed",
        "Fraudster performed SIM swap at telecom. Intercepted OTP for high-value transfer. "
        "Transaction from new device + new location. V2, V8 deviated strongly.",
        "very_high", "any",
        "SIM swap confirmed with telecom. Transfer clawed back."),

    FraudCase("ATO-005", "account_takeover",
        "Dormant account reactivated — insider fraud",
        "Dormant account with no activity for 6 months suddenly had large transfer. "
        "No prior rolling activity. rolling_count_24h = 0 then sudden spike. "
        "V14 and V21 showed anomaly.",
        "high", "day",
        "Flagged as dormant account activity. Investigated as potential insider fraud."),

    # --- Velocity / Rapid Fire ---
    FraudCase("VEL-001", "velocity",
        "20 transactions in 1 hour — card cloned",
        "Cloned card used at petrol stations and ATMs rapidly. "
        "20 transactions in under 60 minutes. rolling_count_1h = 20. "
        "All amounts were round numbers (ATM pattern).",
        "medium", "any",
        "Velocity alert fired after 8th transaction. Card blocked."),

    FraudCase("VEL-002", "velocity",
        "Rapid restaurant + retail sequence — gang fraud",
        "Organised gang cloned multiple cards, used simultaneously at different POS terminals. "
        "Each card showed high 1h velocity. Coordinated across 4 cities.",
        "medium", "night",
        "Network analysis detected coordinated attack. 47 cards affected."),

    FraudCase("VEL-003", "velocity",
        "E-commerce burst — compromised merchant DB",
        "Merchant database breached. Cards used in burst within 6 hours of breach. "
        "Very high rolling_count_24h for affected cards. V9 strongly deviated.",
        "medium", "any",
        "Merchant informed. Bulk card replacement initiated."),

    FraudCase("VEL-004", "velocity",
        "ATM daily limit repeated — card sharing",
        "Same card used at 3 different ATMs to extract daily maximum. "
        "Possible card shared among multiple people or multiple counterfeit copies.",
        "high", "night",
        "Daily limit pattern flagged. Card cancelled and reissued."),

    FraudCase("VEL-005", "velocity",
        "Micro-transaction burst before large purchase",
        "Classic pattern: 10 transactions under $2 to test card, then $3000 purchase. "
        "rolling_count_1h = 11. amount_log jumped sharply for final transaction.",
        "very_high", "night",
        "Large transaction blocked. Prior micro-transactions confirmed testing pattern."),

    # --- High Amount Anomaly ---
    FraudCase("HAM-001", "high_amount",
        "Single 10x transaction — mule account transfer",
        "Account used as mule. Received large transfer then immediately withdrew. "
        "Single transaction 10x above 30-day average. V14 extreme deviation.",
        "very_high", "any",
        "Mule account identified. Funds frozen. Linked to organised crime network."),

    FraudCase("HAM-002", "high_amount",
        "Jewellery store high purchase — physically stolen card",
        "Card stolen from handbag. Used at high-end jewellery store within 30 minutes. "
        "is_night = False but amount was 8x normal. V4, V11 deviated.",
        "very_high", "day",
        "Cardholder reported theft. CCTV footage obtained from merchant."),

    FraudCase("HAM-003", "high_amount",
        "Electronics store bulk purchase — resale fraud",
        "Multiple high-value electronics purchased using stolen card details. "
        "Common resale fraud pattern. Purchases at multiple stores in sequence. "
        "rolling_amount_24h very high.",
        "very_high", "day",
        "Merchant alerted. Items recovered at pawn shop."),

    FraudCase("HAM-004", "high_amount",
        "Crypto exchange large buy — card not present",
        "Stolen card used to buy cryptocurrency immediately convertible to cash. "
        "Irreversible transaction. V3 and V17 deviated. Night-time transaction.",
        "very_high", "night",
        "Exchange alerted within 90 seconds. Funds frozen before withdrawal."),

    FraudCase("HAM-005", "high_amount",
        "Wire transfer to foreign account — BEC fraud",
        "Business email compromise. CFO impersonated via email. Employee initiated "
        "large wire transfer. Single transaction significantly above normal. "
        "V1, V5, V14 all showed deviations.",
        "very_high", "day",
        "Transfer reversed after SWIFT recall. Legal proceedings initiated."),

    # --- Night / Off-hours ---
    FraudCase("NOH-001", "night_fraud",
        "3AM online purchase — stolen card after mugging",
        "Victim mugged at night. Card used online within 20 minutes. "
        "is_night = True. Amount 6x above average. V14 strongly deviated.",
        "high", "night",
        "Victim called within 30 min. Card blocked. Purchase reversed."),

    FraudCase("NOH-002", "night_fraud",
        "Late night fuel + fast food + ATM — card skimmed at pump",
        "Card skimmed at petrol station. Clone used late night for small purchases. "
        "Pattern: fuel → fast food → ATM. rolling_count_1h = 3 with increasing amounts.",
        "medium", "night",
        "Skimmer found at pump station. Multiple victims identified."),

    FraudCase("NOH-003", "night_fraud",
        "Overseas online transaction while cardholder asleep — credential theft",
        "Card credentials sold on dark web. Used by overseas buyer at 4AM local time. "
        "High amount, night-time, V3 deviation.",
        "high", "night",
        "Geo-IP mismatch flagged. Transaction blocked automatically."),

    FraudCase("NOH-004", "night_fraud",
        "Gaming platform credits purchased — child/teen account compromise",
        "Minor's account accessed at 2AM. In-game currency purchases. "
        "Low amounts but high frequency. is_night = True. V20 deviated.",
        "low", "night",
        "Parent alerted. Account access restricted. Refund issued."),

    FraudCase("NOH-005", "night_fraud",
        "Pharmacy purchase at 1AM — prescription fraud",
        "Card used at 24h pharmacy for controlled substances. "
        "Unusual for cardholder. is_night = True. V7 deviated.",
        "medium", "night",
        "Flagged for review. Pharmacist contacted. Prescription was fraudulent."),

    # --- PCA Feature Anomaly (V1-V28) ---
    FraudCase("PCA-001", "pca_anomaly",
        "Strong V14 deviation — known fraud signature",
        "V14 is one of the strongest fraud predictors in this dataset. "
        "Extreme negative V14 value combined with high amount is a strong fraud signal. "
        "Frequently seen in CNP and ATO fraud patterns.",
        "high", "any",
        "V14 threshold breach. High precision fraud signal historically."),

    FraudCase("PCA-002", "pca_anomaly",
        "V10 and V12 combined deviation — merchant category anomaly",
        "V10 and V12 deviating together often indicates merchant category mismatch. "
        "Card normally used for grocery/fuel but transaction at electronics/luxury. "
        "Combined deviation is rare in legitimate transactions.",
        "medium", "any",
        "Merchant category anomaly confirmed by cardholder."),

    FraudCase("PCA-003", "pca_anomaly",
        "V4 deviation — time-of-day anomaly pattern",
        "V4 captures time-related PCA component. Strong deviation suggests "
        "transaction occurring at unusual time for this cardholder's behaviour profile.",
        "medium", "night",
        "Behavioural time pattern confirmed as anomalous."),

    FraudCase("PCA-004", "pca_anomaly",
        "V17 extreme deviation — high-risk merchant type",
        "V17 deviation linked to specific high-risk merchant categories. "
        "Combined with high amount and night context creates strong fraud signal. "
        "Seen frequently in digital goods fraud.",
        "high", "night",
        "High-risk merchant category confirmed. Real-time block applied."),

    FraudCase("PCA-005", "pca_anomaly",
        "Multiple V features deviating — synthetic identity fraud",
        "5+ PCA features deviating simultaneously. Rare in genuine transactions. "
        "Often seen in synthetic identity fraud where no real behavioural baseline exists. "
        "V1, V3, V4, V10, V14 all deviated.",
        "very_high", "any",
        "Synthetic identity confirmed. Account closed. Law enforcement notified."),

    # --- Low Amount But Flagged ---
    FraudCase("LAF-001", "low_amount_flag",
        "Subscription trial abuse — stolen card for free trials",
        "Stolen card used to sign up for free trials requiring card-on-file. "
        "Amounts are $1-$5 (verification charges). V20 and rolling_count_24h elevated.",
        "low", "any",
        "Pattern matched free trial abuse. Card blocked after 8th sign-up."),

    FraudCase("LAF-002", "low_amount_flag",
        "Charity donation fraud — card testing via charities",
        "Fraudster donated $1 to multiple charities to test card validity before "
        "large purchase. High frequency, very low amounts. known testing pattern.",
        "low", "day",
        "Charity platforms alerted. Card blocked before main fraud transaction."),

    # --- Weekend Fraud ---
    FraudCase("WKD-001", "weekend",
        "Weekend shopping spree — card stolen at event",
        "Card stolen at public event on weekend. Used at multiple retail stores in 2h. "
        "is_weekend = True. rolling_amount_2h unusually high. V4 deviated.",
        "high", "weekend",
        "Card blocked after 4th transaction. Victim confirmed theft at event."),

    FraudCase("WKD-002", "weekend",
        "Weekend night club / entertainment fraud — card skimmed",
        "Card skimmed at night club POS terminal. Used multiple times at same venue "
        "and nearby ATM. is_weekend = True. is_night = True. V9 deviated.",
        "medium", "weekend",
        "Skimmer found at venue. Several victims. Card network alerted."),

    # --- Dormant Account ---
    FraudCase("DRM-001", "dormant",
        "6-month dormant account — sudden large transaction",
        "Account inactive for 180+ days. Sudden large transaction with no prior "
        "rolling activity. rolling_count_24h = 0 → immediate large amount. "
        "V14 and V5 strongly deviated.",
        "very_high", "any",
        "Dormant account flag triggered. Cardholder confirmed account compromised."),

    FraudCase("DRM-002", "dormant",
        "Forgotten card reactivated — dark web data",
        "Old card credentials found on dark web after old breach. "
        "Card last used 1 year ago. Sudden high-value online purchase. "
        "No rolling baseline — all velocity features return 0.",
        "high", "night",
        "Zero rolling baseline combined with high amount flagged immediately."),
]


def get_all_cases() -> List[FraudCase]:
    """Return all fraud case documents."""
    return FRAUD_CASES


def get_case_texts() -> List[str]:
    """Return list of full text descriptions for embedding."""
    return [
        f"{case.title}. {case.description} Outcome: {case.outcome}"
        for case in FRAUD_CASES
    ]


def get_case_ids() -> List[str]:
    """Return list of case IDs for ChromaDB document IDs."""
    return [case.case_id for case in FRAUD_CASES]


def get_case_metadata() -> List[dict]:
    """Return metadata dicts for ChromaDB filtering."""
    return [
        {
            "case_id":     case.case_id,
            "pattern":     case.pattern,
            "amount_range": case.amount_range,
            "time_context": case.time_context,
        }
        for case in FRAUD_CASES
    ]
