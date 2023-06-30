import re


outputs = [
    "Human 1 A: 1.Nh3 Nc6 2.e4 Nf6 3.a4 e6 4.Nc3 d5 5.exd5 exd5 6.Nf4 Bd6 7.Nfxd5 Nxd5 8.Nxd5 Be6 9.Ne3 O-O 10.Bb5 Qf6 11.O-O a6 12.Bxc6 bxc6 13.d3 Qg6 14.Nc4 Bh3 15.Nxd6 cxd6 16.g Human 1 A: 3.Qf3 is a bad move. You should play 3.Qh5. Human 0 A: Quote: Human 1 wrote:",
    "Human 1 A: 1. Nf3 Nf6 2. Nxe5 Nxe4 3. Nxf7 Nxc3 4. Nxd8 Nxd1 5. Nxb7 Nc3 6. Nxd8 Nxb1 7. Nc6 Nc3 8. Nxa7 Nxa2 9. Nxc8 Nxc1 10. Nxe7 Nxe2 11. Nxg8 Nxg1 12. Nxh6 Nxh3 13. gxh3 gxh6 14. Bg2 Bxd2+ 15. Kxd2 O-O-O 16. Kc3 Rhe8 17. Kb2 Re2+ 18. Kc1 Rxf2 19. Kb1 Rxg2 20. Kc1 Rxh2 21. Kb1 Rxh1+ 22. Kb2 Rxh4 23. Kc3 Rf4 24"
]

def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))

print(outputs[1])
splitted = outputs[1].replace(".", " ").replace(":", " ").split(" ")
output = [x for x in splitted if has_numbers(x) and len(x) > 1 and len(x) <= 5 and not x.isnumeric()]

print(output)