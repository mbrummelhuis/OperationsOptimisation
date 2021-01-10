\ Model runway_allocation
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  808.356 x00 + 1495.967 x01 + 1602.731 x02 + 901.139 x10 + 1321.84 x11
   + 987.567 x12 + 808.356 x20 + 1495.967 x21 + 1602.731 x22 + 1495.967 x30
   + 639.313 x31 + 973.586 x32 + 901.139 x40 + 1321.84 x41 + 987.567 x42
   + 24 G0 + 24 G1 + 24 G2
Subject To
 Flight_f0: x00 + x01 + x02 = 1
 Flight_f1: x10 + x11 + x12 = 1
 Flight_f2: x20 + x21 + x22 = 1
 Flight_f3: x30 + x31 + x32 = 1
 Flight_f4: x40 + x41 + x42 = 1
 WR_f0_r2: x02 = 1
 WR_f1_r0: x10 = 1
 WR_f1_r1: x11 = 1
 WR_f2_r2: x22 = 1
 WR_f3_r0: x30 = 1
 WR_f4_r1: x41 = 1
 AC_sep_f0_f1_r0: 820 x40 >= 120
 AC_sep_f0_f1_r1: 799 x41 >= 120
 AC_sep_f0_f1_r2: 914 x42 >= 120
 AC_sep_f0_f2_r0: 1661 x40 >= 120
 AC_sep_f0_f2_r1: 1640 x41 >= 120
 AC_sep_f0_f2_r2: 1755 x42 >= 120
 AC_sep_f0_f3_r0: 2385 x40 >= 120
 AC_sep_f0_f3_r1: 2364 x41 >= 120
 AC_sep_f0_f3_r2: 2479 x42 >= 120
 AC_sep_f0_f4_r0: 3167 x40 >= 120
 AC_sep_f0_f4_r1: 3146 x41 >= 120
 AC_sep_f0_f4_r2: 3261 x42 >= 120
 AC_sep_f1_f0_r0: - 700 x40 >= 120
 AC_sep_f1_f0_r1: - 721 x41 >= 120
 AC_sep_f1_f0_r2: - 606 x42 >= 120
 AC_sep_f1_f2_r0: 901 x40 >= 120
 AC_sep_f1_f2_r1: 880 x41 >= 120
 AC_sep_f1_f2_r2: 995 x42 >= 120
 AC_sep_f1_f3_r0: 1625 x40 >= 120
 AC_sep_f1_f3_r1: 1604 x41 >= 120
 AC_sep_f1_f3_r2: 1719 x42 >= 120
 AC_sep_f1_f4_r0: 2407 x40 >= 120
 AC_sep_f1_f4_r1: 2386 x41 >= 120
 AC_sep_f1_f4_r2: 2501 x42 >= 120
 AC_sep_f2_f0_r0: - 1541 x40 >= 120
 AC_sep_f2_f0_r1: - 1562 x41 >= 120
 AC_sep_f2_f0_r2: - 1447 x42 >= 120
 AC_sep_f2_f1_r0: - 781 x40 >= 120
 AC_sep_f2_f1_r1: - 802 x41 >= 120
 AC_sep_f2_f1_r2: - 687 x42 >= 120
 AC_sep_f2_f3_r0: 784 x40 >= 120
 AC_sep_f2_f3_r1: 763 x41 >= 120
 AC_sep_f2_f3_r2: 878 x42 >= 120
 AC_sep_f2_f4_r0: 1566 x40 >= 120
 AC_sep_f2_f4_r1: 1545 x41 >= 120
 AC_sep_f2_f4_r2: 1660 x42 >= 120
 AC_sep_f3_f0_r0: - 2265 x40 >= 120
 AC_sep_f3_f0_r1: - 2286 x41 >= 120
 AC_sep_f3_f0_r2: - 2171 x42 >= 120
 AC_sep_f3_f1_r0: - 1505 x40 >= 120
 AC_sep_f3_f1_r1: - 1526 x41 >= 120
 AC_sep_f3_f1_r2: - 1411 x42 >= 120
 AC_sep_f3_f2_r0: - 664 x40 >= 120
 AC_sep_f3_f2_r1: - 685 x41 >= 120
 AC_sep_f3_f2_r2: - 570 x42 >= 120
 AC_sep_f3_f4_r0: 842 x40 >= 120
 AC_sep_f3_f4_r1: 821 x41 >= 120
 AC_sep_f3_f4_r2: 936 x42 >= 120
 AC_sep_f4_f0_r0: - 3047 x40 >= 120
 AC_sep_f4_f0_r1: - 3068 x41 >= 120
 AC_sep_f4_f0_r2: - 2953 x42 >= 120
 AC_sep_f4_f1_r0: - 2287 x40 >= 120
 AC_sep_f4_f1_r1: - 2308 x41 >= 120
 AC_sep_f4_f1_r2: - 2193 x42 >= 120
 AC_sep_f4_f2_r0: - 1446 x40 >= 120
 AC_sep_f4_f2_r1: - 1467 x41 >= 120
 AC_sep_f4_f2_r2: - 1352 x42 >= 120
 AC_sep_f4_f3_r0: - 722 x40 >= 120
 AC_sep_f4_f3_r1: - 743 x41 >= 120
 AC_sep_f4_f3_r2: - 628 x42 >= 120
 NLSC_f0_r0_G0: 14500 x00 - 100000 G0 <= 4000
 NLSC_f0_r1_G1: 700 x01 - 100000 G1 <= 4000
 NLSC_f0_r2_G2: 13600 x02 - 100000 G2 <= 4000
 NLSC_f1_r0_G0: 14500 x10 - 100000 G0 <= 4000
 NLSC_f1_r1_G1: 700 x11 - 100000 G1 <= 4000
 NLSC_f1_r2_G2: 13600 x12 - 100000 G2 <= 4000
 NLSC_f2_r0_G0: 14500 x20 - 100000 G0 <= 4000
 NLSC_f2_r1_G1: 700 x21 - 100000 G1 <= 4000
 NLSC_f2_r2_G2: 13600 x22 - 100000 G2 <= 4000
 NLSC_f3_r0_G0: 14500 x30 - 100000 G0 <= 4000
 NLSC_f3_r1_G1: 700 x31 - 100000 G1 <= 4000
 NLSC_f3_r2_G2: 13600 x32 - 100000 G2 <= 4000
 NLSC_f4_r0_G0: 14500 x40 - 100000 G0 <= 4000
 NLSC_f4_r1_G1: 700 x41 - 100000 G1 <= 4000
 NLSC_f4_r2_G2: 13600 x42 - 100000 G2 <= 4000
Bounds
 x00 = 0
 x01 = 0
 x02 = 0
 x10 = 0
 x11 = 0
 x12 = 0
 x20 = 0
 x21 = 0
 x22 = 0
 x30 = 0
 x31 = 0
 x32 = 0
 x40 = 0
 x41 = 0
 x42 = 0
Binaries
 x00 x01 x02 x10 x11 x12 x20 x21 x22 x30 x31 x32 x40 x41 x42 G0 G1 G2
End
