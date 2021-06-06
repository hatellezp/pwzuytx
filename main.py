from prime import *
from quadratic_sieve import BPrimes

limit = 2546789432234567898765445678765432456787654334567898765432234123453245679087654456789876543456789765488765476543234567898797890879765432134567890987654321344697654321897654334567876543245678765432345678765432

for n in range(limit, limit + 2):
    k = n // 2

    is_prime = linear_primality_test(n)
    solovay = solovay_strassen_test(n, k)
    miller = miller_rabin_test(n, k)

    print(f"--- testing value: {n}")
    if (is_prime and not solovay) or (not is_prime and solovay):
        # print("-" * 50)
        print(f"  n: {n} is prime: {is_prime}\n  solovay: {solovay}\n  miller: {miller}")
    elif (is_prime and not miller) or (not is_prime and miller):
        # print("-" * 50)
        print(f"  n: {n} is prime: {is_prime}\n  solovay: {solovay}\n  miller: {miller}")

""" 
for n in range(2, 100):
    k = n // 2

    is_prime = miller_rabin_test(n, k)

    if not is_prime:
        factor = pollard_rho_naive(n)
        accum = 0
        while factor == -1 and accum < 10:
            accum += 1
            factor = pollard_rho_naive(n)

        if factor == -1:
            print(f" failed here !!! n: {n}, tried {accum} times and factor is {factor}")
        # print(f"n: {n}\n  is prime: {linear_primality_test(n)}\n  factor: {pollard_rho_naive(n)}")
    else:
        # print(f"n: {n}\n  is prime: {linear_primality_test(n)}\n  factor: {pollard_rho_naive(n)}")
        pass


for n in range(2, 100):
    print(f"n: {n}, factors: {naive_factorize_rho_method(n)}")

"""

# n = 1311097532562595991877980619849724606784164430105441327897358800116889057763413423
# n = 76543543675437654312123298
n = 15347 * pow(103*149, 30)
old_n = n

n = n // pow(149, 31)
n = n // pow(103, 23)

# n = 83 * pow(7, 12) * pow(29, 3) * pow(2, 222) * pow(101, 30) * pow(17, 10)
print(len("1311097532562595991877980619849724606784164430105441327897358800116889057763413423"))
# factors = naive_factorize_rho_method(n, rho_limit=10, rho_tries=10)

# print(factors)
# print(verify_factorization(n, factors))

# factors = {149:31, 103: 31}
# print(verify_factorization(old_n, factors))



bprimes = BPrimes(5)
n = 25 * 16 * 97
rem, fs = bprimes.factorize(n)

print(bprimes)
print(rem, fs)

print(n == rem * bprimes.aggregate(fs))


bprimes = BPrimes(ps=[2, 5, 97])
n = 25 * 16 * 97
rem, fs = bprimes.factorize(n)

print(bprimes)
print(rem, fs)

print(n == rem * bprimes.aggregate(fs))


