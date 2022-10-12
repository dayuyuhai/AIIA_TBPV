ans = 0
def guess(n, cnt):
    while n > 1:
        cnt += 1
        if n % 2:
            if n != 1:
                n = n * 3 + 1
        else:
            n //= 2
    return cnt


for i in range(1, 101):
    cnt = guess(i, 1)
    print(i, " length ", cnt)
    if cnt > ans:
        ans = cnt
    
print("ans = ", ans)