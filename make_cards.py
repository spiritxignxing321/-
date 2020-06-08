# -*-coding:utf-8-*-

# (1) BUILD CARDS THAT ARE POSSIBLE HANDOUT FOR DOU DI ZHU
# (2) BUILD SAMPLES FROM EXCEL DATA

# YISONG WANG
# 2019.09.25

import numpy as np
from pandas import DataFrame
import itertools
import xlrd
import argparse

'''
出牌类型    可能出现的个数    Label表示的位置
PASS    1    0
单牌        15    1-15
对子        13    16-28
3连对    10    29-38
4连对    9    39-47
5连对    8    48-55
6连对    7    56-62
7连对    6    63-68
8连对    5    69-73
9连对    4    74-77
10连对    3    78-80
5连牌    8    81-88
6连牌    7    89-95
7连牌    6    96-101
8连牌    5    102-106
9连牌    4    107-110
10连牌    3    111-113
11连牌    2    114-115
12连牌    1    116
炸弹（4张牌）    13    117-129
王炸        1    130
3个牌单出        13    131-143
飞机不带牌（2连）    11    144-154
飞机不带牌（3连）    10    155-164
飞机不带牌（4连）    9    165-173
飞机不带牌（5连）    8    174-181
飞机不带牌（6连）    7    182-188
（以下为带牌情况）        
3带1        13    189-201
飞机带牌（2连）    11    202-212
飞机带牌（3连）    10    213-222
飞机带牌（4连）    9    223-231
飞机带牌（5连）    8    232-239
4带2（两单牌）    13    240-252
4带2（对子）        13    253-265

3-10用数字3-10表示，J,Q,K,A,2,小王,大王分别用11,12,13,14,15,16,17表示
'''

cards_type = {'pass', 'solo', 'pair', 'trio', 'bomb', 'nuke', 'solo-chain', \
              'pair-chain', 'trio-chain', 'trio-[chain]-solo', 'trio-[chain]-pair', 'quad-solo2', 'quad-pairs2'}

dict_principle_type = {0: 'pass', \
                       1: 'solo', \
                       2: 'pair', \
                       3: 'trio', \
                       4: 'bomb', \
                       5: 'nuke', \
                       6: 'solo-chain', \
                       7: 'pair-chain', \
                       8: 'trio-chain', \
                       9: 'trio-[chain]-solo', \
                       10: 'trio-[chain]-pair', \
                       11: 'quad-solo2', \
                       12: 'quad-pairs2'}

Cards = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
Cards_no_jokers = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def build_principle_type():
    dict_p_type = {}
    for i in range(len(cards_type)):
        dict_p_type[i] = cards_type[i]
    return dict_p_type


def build_actions():
    dict_cards = {}
    key = 0
    simple_key = 0
    # Build the Dou Di Zhu cards in the form of 

    #### ID  cards    principle_type_key principle_value simple_key

    # cards: the cards in the handout
    # type: {'pass', 'solo', 'pair', 'trio', 'bomb', 'nuke', 'solo-chain', 'pair-chain', 'trio-chain', 'trio-[chain]-solo', 'trio-[chain]-pair', 'bomb-solo2', 'bomb-pairs2'}
    # the principle value for each type is repectively: principle_value(type) which is the least card forms a cards_type
    # e.g., principle_value(555666??)==5, principle_value(333?)==3, principle_value(pass) == 0
    #       where '?' is an arbitrary solo card or a pair
    # simple_key: the key that ignore the trio with additional card, i.e., 3334 and 3335 have the same key
    # and write into an excel file

    # (1) Pass
    # 0    [0]
    dict_cards[key] = [[], 0, 0, simple_key]
    key += 1
    simple_key += 1
    # (2) solo
    # 3-17, 
    for i in range(3, 18):
        dict_cards[key] = [[i], 1, i, simple_key]
        key += 1
        simple_key += 1
    # (3) pair: one pair
    for i in range(3, 16):
        dict_cards[key] = [[i, i], 1, i, simple_key]
        key += 1
        simple_key += 1

    # (4) pair-chain: 3-10 pairs
    for i in range(3, 11):  # len(dict_cards[key])==2*i
        for j in range(3, 16 - i):  # the cards in the pairs, always starting from 3 to 14 (Ace)
            value = list()
            for k in range(j, j + i):
                value.extend([k, k])
            dict_cards[key] = [value, 7, j, simple_key]
            key += 1
            simple_key += 1

    # (5) 5-12 solo-chains
    for i in range(5, 13):  # len(dict_cards[key])==i
        for j in range(3, 16 - i):  # the cards in the solos, always starting from 3 to 14 (Ace)
            value = list()
            for k in range(j, j + i):
                value.extend([k])
            dict_cards[key] = [value, 6, j, simple_key]
            key += 1
            simple_key += 1

    # (6) Bomb with 4 cards
    for i in range(3, 16):  # the cards in the bomb
        dict_cards[key] = [[i, i, i, i], 4, i, simple_key]
        key += 1
        simple_key += 1
    # Bomb with 2 jokers
    dict_cards[key] = [[16, 17], 4, 16, simple_key]
    key += 1
    simple_key += 1

    # (7) Trio
    for j in range(3, 16):  # the cards in the trios, always starting from 3 to 15
        dict_cards[key] = [[j, j, j], 3, j, simple_key]
        key += 1
        simple_key += 1
    # Trio-chain    2-6 Trio-chain without any more cards
    for i in range(2, 7):  # len(dict_cards[key])==3*i
        for j in range(3, 16 - i):  # the cards in the trios, always starting from 3 to 14 (Ace)
            value = list()
            for k in range(j, j + i):
                value.extend([k, k, k])
            dict_cards[key] = [value, 8, j, simple_key]
            key += 1
            simple_key += 1

    # (8) 1-5 Trio with one more card in each Trio
    # Trio-[chain]-solo
    for j in range(3, 16):  # the cards in the trios, always starting from 3 to 14 (Ace) and  2
        for cm in sorted(list(set(Cards) - {j})):
            dict_cards[key] = [[j, j, j] + [cm], 9, j, simple_key]
            key += 1
        simple_key += 1

    for i in range(2, 6):  # len(dict_cards[key])==4*i
        for j in range(3, 16 - i):  # the cards in the trios, always starting from 3 to 14 (Ace)
            value = list()
            for k in range(j, j + i):
                value.extend([k, k, k])  # all the Trios
            candidates = sorted(list(set(Cards) - set(value)) + list(set(Cards_no_jokers) - set(value)))
            for cm in itertools.combinations(candidates, i):  # all the one more card for each Trio
                if {16, 17} <= set(cm): continue
                dict_cards[key] = [value + list(cm), 9, j, simple_key]
                key += 1
            simple_key += 1

    # (9) 1-5 Trio with one more pair in each Trio
    # Trio-[chain]-pair
    for j in range(3, 16):  # the cards in the trios, always starting from 3 to 14 (Ace) and  2
        for cm in sorted(list(set(Cards_no_jokers) - {j})):
            dict_cards[key] = [[j, j, j] + [cm, cm], 10, j, simple_key]
            key += 1
        simple_key += 1

    for i in range(2, 4):  # len(dict_cards[key])==5*i
        for j in range(3, 16 - i):  # the cards in the trios, always starting from 3 to 14 (Ace)
            value = list()
            for k in range(j, j + i):
                value.extend([k, k, k])  # all the Trios
            for cm in itertools.combinations(set(Cards_no_jokers) - set(value),
                                             i):  # all the one more pair for each Trio
                dict_cards[key] = [value + sorted(list(cm) + list(cm)), 9, j, simple_key]
                key += 1
            simple_key += 1
    # (9) Quad with two more cards
    for i in range(3, 16):
        value = [i, i, i, i]
        candidates = sorted(list(set(Cards) - {i}) + list(set(Cards_no_jokers) - {i}))
        for cm in itertools.combinations(candidates, 2):  # all the two more cards
            if {16, 17} <= set(cm): continue
            dict_cards[key] = [value + list(cm), 10, i, simple_key]
            key += 1
        simple_key += 1

    # (10) Quad with two more pairs 
    for i in range(3, 16):
        value = [i, i, i, i]
        for cm in itertools.combinations(set(Cards_no_jokers) - {i}, 2):  # all the two more cards
            dict_cards[key] = [value + sorted(list(cm) + list(cm)), 11, i, simple_key]
            key += 1
        simple_key += 1

    return (dict_cards)


def list_2_str(lst):
    return ','.join([str(x) for x in lst])


def cards_to_excel(file_name):
    cards_actions = build_actions()

    # print(cards_actions)
    workbook = xlsxwriter.Workbook(file_name)
    worksheet_type = workbook.add_worksheet("Principle type")

    for i in range(len(dict_principle_type)):
        if i in dict_principle_type:
            worksheet_type.write(i, 0, i)
            worksheet_type.write(i, 1, dict_principle_type[i])

    worksheet = workbook.add_worksheet("Keys-Cards")
    row = 1
    worksheet.write(0, 0, "key")
    worksheet.write(0, 1, "cards")
    worksheet.write(0, 2, "principle_type")
    worksheet.write(0, 3, "principle_value")
    worksheet.write(0, 4, "simple_key")

    for i in range(len(cards_actions)):
        if i in cards_actions:
            worksheet.write(row, 0, i)
            worksheet.write(row, 1, list_2_str(cards_actions.get(i)[0]))
            worksheet.write(row, 2, cards_actions.get(i)[1])
            worksheet.write(row, 3, cards_actions.get(i)[2])
            worksheet.write(row, 4, cards_actions.get(i)[3])
            row += 1
    workbook.close()


'''
    worksheet_value = workbook.add_worksheet("Cards-Keys")
    row = 1
    worksheet.write(0, 0, "Card")
    worksheet.write(0, 1, "Key")
    for i in range(len(cards_actions)):
        if i in cards_actions:
            worksheet.write(row, 1, i)
            worksheet.write(row, 0, list_2_str(cards_actions.get(i)[0])) 
            row += 1
'''

'''
MatchId	Identity	IsFirst	PlayCrads	BCrads	UpPlayCrads	UpPutCrads	DownPlayCrads	DownPutCrads	PlayNo
1	1	1	3,3,3,7,7	17,16,15,15,14,12,12,10,10,7,7,6,6,6,3,3,3,11,14,14					1
1	0	0		9,12,4,6,5,8,15,4,7,13,11,3,11,4,10,13,10	3,3,3,7,7	3,3,3,7,7			2
1	0	0	5,5,5,13,13	12,4,14,8,13,5,8,15,11,7,9,13,5,9,9,8,5			3,3,3,7,7	3,3,3,7,7	3
1	1	0	6,6,6,10,10	17,16,15,15,14,14,14,12,12,11,10,10,6,6,6	5,5,5,13,13	5,5,5,13,13			4
1	0	0		15,13,13,12,11,11,10,10,9,8,7,6,5,4,4,4,3	6,6,6,10,10	3,3,3,7,7,6,6,6,10,10	5,5,5,13,13	5,5,5,13,13	5
1	0	0		15,14,12,11,9,9,9,8,8,8,7,4			6,6,6,10,10	3,3,3,7,7,6,6,6,10,10	6
1	1	1	12,12,14,14,14	17,16,15,15,14,14,14,12,12,11		5,5,5,13,13			7
1	0	0		15,13,13,12,11,11,10,10,9,8,7,6,5,4,4,4,3	12,12,14,14,14	3,3,3,7,7,6,6,6,10,10,12,12,14,14,14		5,5,5,13,13	8
1	0	0		15,14,12,11,9,9,9,8,8,8,7,4			12,12,14,14,14	3,3,3,7,7,6,6,6,10,10,12,12,14,14,14	9
1	1	1	15,15	17,16,15,15,11		5,5,5,13,13			10
1	0	0		15,13,13,12,11,11,10,10,9,8,7,6,5,4,4,4,3	15,15	3,3,3,7,7,6,6,6,10,10,12,12,14,14,14,15,15		5,5,5,13,13	11
1	0	0		15,14,12,11,9,9,9,8,8,8,7,4			15,15	3,3,3,7,7,6,6,6,10,10,12,12,14,14,14,15,15	12
1	1	1	16,17	17,16,11		5,5,5,13,13			13
1	0	0		15,13,13,12,11,11,10,10,9,8,7,6,5,4,4,4,3	16,17	3,3,3,7,7,6,6,6,10,10,12,12,14,14,14,15,15,16,17		5,5,5,13,13	14
1	0	0		15,14,12,11,9,9,9,8,8,8,7,4			16,17	3,3,3,7,7,6,6,6,10,10,12,12,14,14,14,15,15,16,17	15
1	1	1	11	11		5,5,5,13,13			16

SAMPLE DATA ARE IN FORM OF:
SAMPLE: 16*6 INTEGER
16×6：integer
（a）Row 0：本玩家身份编码 {0,1,2} 
（b）Row 1-15：扑克（3-17）的个数(当为未轮到出牌时填充为-1，全为0时表示pass动作)
    a)Col 1：当前玩家手牌
    b)Col 2：当前玩家出过的所有牌
    c)Col 3：上家出过的所有牌
    d)Col 4：下家出过的所有牌
    e)Col 5：本轮上家出的牌
    f)Col 6：本轮下家出的牌
LABEL: THE CORRSPONDING CARDS ID 
'''


# build sample from the history in the above format

def cards_str_list(cards_str):  # '5,5,13,13,13' to ordered list [ 5, 5, 13, 13, 13 ]
    if cards_str == '':
        return list()
    else:
        rl = list()
        for x in str(cards_str).split(','):
            if x == '':
                continue
            if float(x) > 16:
                rl.append(int(float(x)) - 1)
            else:
                rl.append(int(float(x)))

        return sorted(rl)


def get_col_cards(cols, col, cards_list):  # assign the cards in cards_list to cols[col,1-15]
    if len(cards_list) != 0 and cards_list[0] == -1:
        for i in range(1, 16):
            cols[i, col] = -1
    else:
        for card in cards_list:
            if card > 16:
                cols[card - 3, col] += 1
            else:
                cols[card - 2, col] += 1


def lists_minus(l1: 'list', l2: 'list'):  # return l1 - l2
    return [x for x in l1 if x not in l2]


def get_label_action(value_key_dict, action_cards, cards_actions_dict, first_level=True):
    # if first_level == true then return the key of the action_cards that discards the non-principle cards, 
    # e.g., the actions 3335 and 3336 have the same key 
    #       33344 and 33399 have the same key but it different from 3334
    #       444456 and 444478 have the same key
    # 
    if not first_level:
        return value_key_dict.get(action_cards)
    else:
        return cards_actions_dict.get(value_key_dict.get(action_cards))[3]  # the simple key


def build_sample(excl_file='data-test.xlsx', to_file_head='sample'):
    # (0) build cards
    cards_actions = build_actions()

    # build the value-key dictionary
    value_key_dict = {}

    for i in range(len(cards_actions)):
        if i in cards_actions:
            temp = cards_actions.get(i)
            key = sorted(cards_actions.get(i)[0])
            value_key_dict[tuple(key)] = i

    # (1) read excl data from excl_file
    workbook = xlrd.open_workbook(filename=excl_file)
    booksheet = workbook.sheet_by_index(0)
    n_samples = 0

    # Data = np.zeros((1)) # the list of samples, 18*6 for one sample
    # Label = list() # the key of the action for the corresponding sample in Data

    # (2) build the sample
    i = 1
    n_samples = 0
    while (i < booksheet.nrows - 2):
        match_ID = booksheet.cell_value(i, 1)  # 对局id
        landlord_ID = booksheet.cell_value(i, 2)  # 本人身份
        down_peasant_ID = booksheet.cell_value(i + 1, 2)  # 下家
        up_peasant_ID = booksheet.cell_value(i + 2, 2)  # 上家身份
        # get the winner id
        j = i + 1
        while (j < booksheet.nrows and booksheet.cell_value(j, 1) == match_ID):
            j += 1
        winner_ID = booksheet.cell_value(j - 1, 2)  # the first player plays out of cards

        if winner_ID == landlord_ID:
            winners = [winner_ID]
        else:
            winners = [down_peasant_ID, up_peasant_ID]

        role_dict = {}
        role_dict[landlord_ID] = 0
        role_dict[down_peasant_ID] = 1
        role_dict[up_peasant_ID] = 2

        k = i
        while (k < j):
            if not booksheet.cell_value(k, 2) in winners:
                k += 1
                continue
            '''
            a)Col 1：当前玩家手牌
            b)Col 2：当前玩家出过的所有牌
            c)Col 3：上家出过的所有牌
            d)Col 4：下家出过的所有牌
            e)Col 5：本轮上家出的牌
            f)Col 6：本轮下家出的牌
            g)Col 7：前一轮我出的牌
            h)Col 8：前一轮上家出的牌
            i)Col 9：前一轮下家出的牌
            j)Col 10：倒数第二轮我出的牌
            k)Col 11：倒数第二轮上家出的牌
            l)Col 12：倒数第二轮下家出的牌
            m)Col 13：倒数第三轮我出的牌
            n)Col 14：倒数第三轮上家出的牌
            o)Col 15：倒数第三轮下家出的牌
            '''
            action_cards = cards_str_list(booksheet.cell_value(k, 4))
            cards = {}

            cards[0] = cards_str_list(booksheet.cell_value(k, 6))  # the current at hand cards of the player
            cards[4] = cards_str_list(booksheet.cell_value(k, 10))  # the current round palyed cards by the up player
            cards[5] = cards_str_list(booksheet.cell_value(k, 15))  # the current round palyed cards by the down player

            cards[1] = lists_minus(cards_str_list(booksheet.cell_value(k, 7)), action_cards)
            # all the palyed cards by the player - the played cards in this round
            # cards[2] = lists_minus(cards_str_list( booksheet.cell_value(k,12) ), cards[4])
            cards[2] = cards_str_list(booksheet.cell_value(k, 12))  # including the current round
            # all the palyed cards by the up player - his played cards in this round
            # cards[3] = lists_minus(cards_str_list( booksheet.cell_value(k,17) ), cards[5])
            cards[3] = cards_str_list(booksheet.cell_value(k, 17))  # including the current round
            # all the palyed cards by the down player - his played cards in this round
            tmp = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

            if k - 3 > 1:
                cards[6] = cards_str_list(booksheet.cell_value(k, 4))
                cards[7] = cards_str_list(booksheet.cell_value(k, 10))
                cards[8] = cards_str_list(booksheet.cell_value(k, 15))
            else:
                cards[6] = cards[7] = cards[8] = tmp

            if k - 6 > 1:
                cards[9] = cards_str_list(booksheet.cell_value(k, 4))
                cards[10] = cards_str_list(booksheet.cell_value(k, 10))
                cards[11] = cards_str_list(booksheet.cell_value(k, 15))
            else:
                cards[9] = cards[10] = cards[11] = tmp

            if k - 9 > 1:
                cards[12] = cards_str_list(booksheet.cell_value(k, 4))
                cards[13] = cards_str_list(booksheet.cell_value(k, 10))
                cards[14] = cards_str_list(booksheet.cell_value(k, 15))
            else:
                cards[12] = cards[13] = cards[14] = tmp

            cols = np.zeros((16, 15))
            cols[0, 0] = role_dict[booksheet.cell_value(k, 2)]  # the role of the player
            cols[0, 6] = cols[0, 9] = cols[0, 12] = cols[0, 1] = cols[0, 0]
            cols[0, 2] = (cols[0, 0] + 2) % 3  # up player role ID
            cols[0, 7] = cols[0, 10] = cols[0, 13] = cols[0, 4] = cols[0, 2]
            cols[0, 3] = (cols[0, 0] + 1) % 3  # down player role ID
            cols[0, 8] = cols[0, 11] = cols[0, 14] = cols[0, 5] = cols[0, 3]

            for col in range(0, 15):
                get_col_cards(cols, col, cards[col])

            if k == i:
                for s in range(1, 16):
                    cols[s, 4] = -1
                    cols[s, 5] = -1
            elif k == i + 1:
                for s in range(1, 16):
                    cols[s, 5] = -1
            # get the sample and its label

            if n_samples == 0:
                Data = cols.copy().reshape((1, 16, 15))
                Label = np.array([get_label_action(value_key_dict, tuple(action_cards), cards_actions)])
                # n_samples += 1
            else:
                try:
                    Label = np.append(Label, [get_label_action(value_key_dict, tuple(action_cards), cards_actions)], 0)
                    Data = np.append(Data, cols.reshape((1, 16, 15)), 0)
                    # n_samples += 1
                except TypeError as reason:
                    print("line %d, the played cards: " % i + str(action_cards))
            k += 1
            # for the next sample
            n_samples += 1

        i = j
    # n_data = np.asarray(Data).reshape((16,6, len(Data)/96))
    # n_label = np.asarray(Label)

    if not to_file_head == "":
        # print(Data[8])
        # print(Data.shape)
        # print(Label)
        # print(Label.shape)
        Data.tofile(to_file_head + "_data")
        Label.tofile(to_file_head + "_label")

    return (Data, Label)


def load_from_file(file_name='sample'):
    # load the data and label from file
    # file_name + '_data'
    # file_name + '_label'
    # repectively
    D = np.fromfile(file_name + '_data')
    L = np.fromfile(file_name + '_label', dtype='int')
    # 构造onehot
    tL = np.zeros((int(D.size / 240), 300), dtype=np.int)
    for x in range(int(D.size / 240)):
        tL[x][L[x]] = 1
    L = tL
    return (D.reshape((int(D.size / 240), 240)), L)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build Dou Di Zhu samples')
    parser.add_argument("file_CNF", help="theory file")
    parser.add_argument("file_M", help="model file")
    # parser.add_argument("-file_S", type = str, help = "S file", required=False) # part of atoms in M

    build_sample("data/ddzmatchdata2.xlsx", "sample")
