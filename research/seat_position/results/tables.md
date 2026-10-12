# Board reconstruction

- 三大家人 2026-09-24 add: screenshot lots found in the feed 10/10, inside rebuilt OI-main top10 10/10, OI mains that day 71, disclosed side-rows on those mains 79
    加空 平安期货 EG2611 +2556
    加空 东方财富期货 BU2611 +2487
    加空 徽商期货 MA2701 +2455
    加多 徽商期货 SA2701 +2438
    加空 平安期货 PP2701 +2041
    加空 平安期货 BU2611 +1555
    加多 徽商期货 FG2701 +1257
    加空 平安期货 FU2611 +1203
    加空 平安期货 MA2701 +1194
    加空 平安期货 EB2611 +1020
- 三大家人 2026-09-24 cut: screenshot lots found in the feed 10/10, inside rebuilt OI-main top10 10/10, OI mains that day 71, disclosed side-rows on those mains 79
    减空 平安期货 FG2701 -2897
    减多 东方财富期货 LH2611 -1238
    减多 徽商期货 SH2611 -1229
    减多 东方财富期货 SH2611 -834
    减空 平安期货 TL2612 -822
    减多 徽商期货 UR2701 -797
    减多 徽商期货 RB2701 -747
    减多 东方财富期货 AO2701 -676
    减多 平安期货 RM2701 -623
    减多 平安期货 C2611 -537
- 四大机构 2026-10-09 add: screenshot lots found in the feed 10/10, inside rebuilt OI-main top10 10/10, OI mains that day 71, disclosed side-rows on those mains 552
    加多 东证期货 RB2701 +26170
    加空 中信期货 MA2701 +18384
    加多 中信期货 MA2701 +16936
    加多 中信期货 M2701 +15389
    加多 东证期货 RM2701 +13638
    加多 东证期货 FG2701 +12820
    加多 东证期货 SA2701 +12594
    加空 国泰君安 MA2701 +10897
    加多 东证期货 M2701 +10745
    加空 东证期货 MA2701 +10365
- 四大机构 2026-10-09 cut: screenshot lots found in the feed 10/10, inside rebuilt OI-main top10 10/10, OI mains that day 71, disclosed side-rows on those mains 552
    减多 中信期货 C2611 -20714
    减空 国泰君安 C2611 -15893
    减空 东证期货 FG2701 -14344
    减多 银河期货 FG2701 -13916
    减空 东证期货 SA2701 -13307
    减空 国泰君安 HC2701 -11152
    减多 东证期货 TA2701 -11005
    减空 银河期货 SA2701 -8388
    减空 东证期货 M2701 -8007
    减空 中信期货 SR2701 -7830
- 三大外资 2026-10-09 add: screenshot lots found in the feed 10/10, inside rebuilt OI-main top10 9/10, OI mains that day 71, disclosed side-rows on those mains 65
    加多 摩根大通 M2701 +10392
    加多 摩根大通 Y2701 +7291
    加空 高盛期货 SM2701 +3875
    加空 高盛期货 HC2701 +3865
    加空 高盛期货 SP2701 +3463
    加多 摩根大通 RM2701 +3015
    加空 高盛期货 RB2701 +2844
    加空 摩根大通 P2701 +2803
    加多 高盛期货 OI2701 +2625
    加空 摩根大通 I2701 +2210
- 三大外资 2026-10-09 cut: screenshot lots found in the feed 10/10, inside rebuilt OI-main top10 10/10, OI mains that day 71, disclosed side-rows on those mains 65
    减多 高盛期货 RB2701 -34661
    减空 高盛期货 SR2701 -17515
    减空 高盛期货 V2701 -13696
    减多 高盛期货 EB2611 -10130
    减多 摩根大通 TA2701 -8294
    减多 高盛期货 M2701 -8028
    减空 高盛期货 RM2701 -7707
    减多 高盛期货 Y2701 -6886
    减多 高盛期货 MA2701 -4962
    减多 摩根大通 P2701 -4243

# Strategies (commodity varieties, gross then net of 1bp per side)

| strategy | 1d mean | 1d t | 1d hit | 1d sharpe | 1d net sharpe | 5d median sharpe | 5d median t |
|---|---:|---:|---:|---:|---:|---:|---:|
| retail_fade_oi | -0.010% | -0.11 | 0.52 | -0.16 | -0.42 | 1.30 | 0.90 |
| retail_follow_oi | 0.010% | 0.11 | 0.48 | 0.16 | -0.09 | -1.30 | -0.90 |
| inst_fade_oi | 0.020% | 0.26 | 0.46 | 0.37 | 0.09 | -0.83 | -0.57 |
| inst_follow_oi | -0.020% | -0.26 | 0.54 | -0.37 | -0.65 | 0.83 | 0.57 |
| foreign_fade_oi | 0.119% | 0.72 | 0.46 | 1.89 | 1.69 | -0.33 | -0.12 |
| foreign_follow_oi | -0.119% | -0.72 | 0.54 | -1.89 | -2.09 | 0.33 | 0.12 |
| retail_fade_raw_lots | 0.013% | 0.16 | 0.55 | 0.22 | -0.02 | 0.28 | 0.19 |
| inst_follow_raw_lots | 0.026% | 0.42 | 0.55 | 0.60 | 0.29 | 0.02 | 0.01 |
| retail_fade_consensus | -5.403% |  | 0.00 |  |  |  |  |
| retail_fade_when_opposite_inst | -0.019% | -0.13 | 0.52 | -0.22 | -0.42 | 0.48 | 0.26 |
| retail_fade_oi_all_assets | -0.047% | -0.61 | 0.49 | -0.87 | -1.15 | 1.06 | 0.73 |
| retail_fade_level | 0.015% | 0.13 | 0.55 | 0.19 | 0.13 | 0.17 | 0.12 |
| inst_follow_level | 0.101% | 1.60 | 0.50 | 2.29 | 2.24 | 0.74 | 0.51 |
| inst_fade_level | -0.101% | -1.60 | 0.50 | -2.29 | -2.34 | -0.74 | -0.51 |

# Cross-sectional Spearman IC vs next open-to-close

- retail: mean IC -0.0017 t -0.09 days 123 share of days IC<0 0.50
- inst: mean IC 0.0140 t 0.83 days 123 share of days IC<0 0.44
- foreign: mean IC 0.0031 t 0.09 days 46 share of days IC<0 0.46
- pingan: mean IC -0.0386 t -1.47 days 115 share of days IC<0 0.57
- dongcai: mean IC -0.0537 t -0.59 days 8 share of days IC<0 0.75
- huishang: mean IC 0.0022 t 0.09 days 120 share of days IC<0 0.49
- retail_raw_lots: mean IC -0.0112 t -0.59 days 123 share of days IC<0 0.55
- retail_level: mean IC -0.0200 t -0.75 days 123 share of days IC<0 0.53
- inst_level: mean IC 0.0057 t 0.39 days 123 share of days IC<0 0.49

# Retail net/OI deciles (0 = most net short add)

- decile 0: mean next open-to-close 0.026% (days 114)
- decile 1: mean next open-to-close -0.058% (days 114)
- decile 2: mean next open-to-close -0.089% (days 114)
- decile 3: mean next open-to-close -0.077% (days 114)
- decile 4: mean next open-to-close 0.055% (days 114)
- decile 5: mean next open-to-close -0.065% (days 114)
- decile 6: mean next open-to-close -0.074% (days 114)
- decile 7: mean next open-to-close -0.040% (days 114)
- decile 8: mean next open-to-close 0.104% (days 114)
- decile 9: mean next open-to-close 0.041% (days 114)
