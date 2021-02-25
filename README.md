# 足球比赛预测

### 足球数据说明

注: 所有数据均为csv格式, 每个csv文件中除了比赛数据, 其他的数据并不会都有

**比赛数据说明**

Div = 赛事类别
Date = 时间 (日/月/年)
Time = 比赛开始时间
HomeTeam = 主场队伍名
AwayTeam = 客场队伍名
FTHG and HG = 主场队伍全场进球数
FTAG and AG = 客场队伍全场进球数
FTR and Res = 全场比赛结果（H =主场胜利，D = 平局，A = 客场胜利）
HTHG = 主场队伍半场进球数
HTAG = 客场队伍半场进球数
HTR = 半场比赛结果（H =主场胜利，D = 平局，A = 客场胜利）



**比赛统计数据说明**

Attendance = 出勤观众
Referee = 比赛裁判
HS = 主场射门数
AS = 客场射门数
HST =主场进球数
AST = 客场进球数
HHW = 主场击中球门框数
AHW = 客场击中球门框数
HC = 主场角球数
AC = 客场角球数
HF = 主场犯规数
AF = 客场犯规数
HFKC = 主场被罚自由球数
AFKC = 客场被罚自由球数
HO =  主场越位数
AO = 客场越位数
HY = 主场黄牌数
AY = 客场黄牌数
HR =主场红牌数
AR = 客场红牌数
HBP = 主场积分 (10 = 黄牌, 25 = 红牌)
ABP = 客场积分 (10 = 黄牌, 25 = 红牌)



**投注赔率数据说明**

B365H = Bet365主队获胜赔率
B365D = Bet365开奖赔率
B365A = Bet365获胜赔率
BSH =蓝色广场主场获胜赔率
BSD =蓝色方块开彩赔率
BSA =蓝色广场客场胜率
BWH = Bet＆Win主队获胜赔率
BWD =赌赢赔率
BWA = Bet＆Win赢赔率
GBH = Gamebookers主场获胜赔率
GBD = Gamebookers赔率
GBA = Gamebookers胜出赔率
IWH = Interwetten主场获胜赔率
IWD = Interwetten开奖赔率
IWA = Interwetten客胜赔率
LBH = Ladbrokes主队获胜赔率
LBD =立博赔率
LBA =立博客赢赔率
PSH和PH =顶峰主胜赔率
PSD和PD =顶峰赔率
PSA和PA =巅峰客胜赔率
SOH =体育赔率主场获胜赔率
SOD =体育赔率
SOA =体育赔率赢赔率
SBH = Sportingbet主场获胜赔率
SBD = Sportingbet开彩赔率
SBA = Sportingbet客胜赔率
SJH = Stan James主场获胜赔率
SJD =斯坦·詹姆斯（Stan James）赔率
SJA =斯坦·詹姆斯客场获胜赔率
SYH =斯坦利贝特主场获胜赔率
SYD =赤柱投注赔率
SYA = Stanleybet赢了赔率
VCH = VC投注主胜赔率
VCD = VC下注赔率
VCA = VC赢走赔率
WHH =威廉希尔主场获胜赔率
WHD = William Hill开奖赔率
WHA =威廉希尔客场赢赔率
Bb1X2 =用于计算比赛赔率平均值和最大值的BetBrain庄家数量
BbMxH = Betbrain最大主场获胜赔率
BbAvH = Betbrain平均主场获胜赔率
BbMxD = Betbrain最大抽奖赔率
BbAvD = Betbrain平均平局获胜几率
BbMxA = Betbrain最大客胜赔率
BbAvA = Betbrain平均客场胜率
MaxH =市场最大主胜赔率
MaxD =市场最大开奖赔率
MaxA =市场最大客胜赔率
AvgH =市场平均主场获胜赔率
AvgD =市场平均开奖赔率
AvgA =市场平均客胜赔率



**总目标投注赔率说明**

BbOU =用于计算超过/低于2.5个目标（总目标）的平均值和最大值的BetBrain庄家数量
BbMx> 2.5 = Betbrain最高超过2.5个进球
BbAv> 2.5 = Betbrain平均超过2.5个目标
BbMx <2.5 =在2.5个目标下最大的Betbrain
BbAv <2.5 = Betbrain平均低于2.5个目标
GB> 2.5 = Gamebookers超过2.5个目标
GB <2.5 =低于2.5个目标的Gamebookers
B365> 2.5 = Bet365超过2.5个目标
B365 <2.5 = Bet365低于2.5个目标
P> 2.5 =巅峰超过2.5个进球
P <2.5 =低于2.5目标的巅峰之作
最高> 2.5 =市场最高超过2.5个目标
最大值<2.5 = 2.5个目标以下的市场最大值
平均> 2.5 =超过2.5个目标的市场平均值
平均<2.5 = 2.5个目标以下的市场平均值



**亚洲让分盘赔率说明**

BbAH =习惯于亚洲残障平均值和最大值的BetBrain庄家数量
BbAHh =障碍的Betbrain大小（主队）
AHh =盘口（主队）的市场规模（自2019/2020起）
BbMxAHH = Betbrain亚洲障碍主队最大赔率
BbAvAHH = Betbrain亚洲残疾人主队平均赔率
BbMxAHA = Betbrain亚洲最大的让分盘球队赔率
BbAvAHA = Betbrain亚洲平均让分盘球队赔率
GBAHH = Gamebookers亚洲障碍主队赔率
GBAHA = Gamebookers亚洲客队客场赔率
GBAH = Gamebookers残障人数（主队）
LBAHH = Ladbrokes亚洲障碍主队赔率
LBAHA = Ladbrokes亚洲让分小组赔率
LBAH = Ladbrokes盘口（主队）
B365AHH = Bet365亚洲让分局主队赔率
B365AHA = Bet365亚洲让分小组赔率
B365AH = Bet365让分盘的大小（主队）
PAHH =巅峰亚洲障碍主队赔率
PAHA =巅峰亚洲客队客场赔率
MaxAHH =市场最大的亚洲让分盘主队赔率
MaxAHA =市场上最大的亚洲让分盘球队赔率	
AvgAHH =亚洲弱势主队市场平均赔率
AvgAHA =市场平均亚洲让分盘球队赔率