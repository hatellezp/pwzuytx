from collections import namedtuple
import random
from typing import Optional, Dict, Tuple
import bisect
import math
from functools import partial

"""
I'm not sure what to do with this yet
"""
PRIMES = {2: 0, 3: 0, 5: 0, 7: 0, 11: 0, 13: 0, 17: 0, 19: 0}

# named tuple to pack information about two integers and their
# relation stated by the Bezout identity
# source: https://en.wikipedia.org/wiki/B%C3%A9zout%27s_identity
Bezout = namedtuple('Bezout', 'acoeff bcoeff gcd lcm')

"""
Power2 pack information about n: 
n = 2**r * d + 1*i
where d is an odd integer and i is zero or one
"""
Power2 = namedtuple("Power2", "n r d i")


def is_perfect_square(n: int) -> Optional[int]:
    """

    :param n: an integer to be tested
    :return: the square root of n is n is a perfect square, None otherwise
    """

    sq = int(math.sqrt(n))

    return sq if n == pow(sq, 2) else None


def equality_modulo(a: int, b: int, mod: int) -> bool:
    """
    verify equality between integers a,b modulo integer n

    :param a: an integer
    :param b: an integer
    :param mod: an integer
    :return: True if a = b (mod mod)
    """
    return (a % mod) == (b % mod)


def decompose_as_power_of_two(n: int) -> Power2:
    """
    see the decription of Power2 namedtuple for
    a description
    """

    # find if n is odd or not and adjust i in consequence
    i = 1 if n % 2 != 0 else 0

    # adjust n to be even
    n -= 1 * i

    # find the biggest power of 2 that divides n
    # TODO: this can be a lot faster with bisection search
    # TODO: bisection is not good because I'm an idiot

    # find r
    r = 0
    while not (n % pow(2, r)):
        r += 1

    r -= 1

    # d is the result of euclidean division
    d = n // pow(2, r)
    n += 1 * i

    return Power2(n, r, d, i)


def extended_euclidean_division(a: int, b: int) -> Bezout:
    """
    there exist two integers such that
        a * acoeff + b * b = greatest_common_divisor(a,b)
    this algorithm return a named tuple with
        - acoeff
        - bcoeff
        - gcd(a,b): greatest_common_divisor(a,b)
        - lcm(a,b): lowest_common_divisor(a,b)

    :param a: an integer
    :param b: an integer
    :return: return the bezout tuple of a,b
    """

    if a < b:
        bez = extended_euclidean_division(b, a)

        # we return the coefficients
        return Bezout(bez.bcoeff, bez.acoeff, bez.gcd, bez.lcm)
    else:
        old_r, r = a, b
        old_s, s = 1, 0
        old_t, t = 0, 1

        # compute euclidean division until we cannot
        while r != 0:
            quotient = old_r // r

            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
            old_t, t = t, old_t - quotient * t

        return Bezout(old_s, old_t, old_r, (a * b) // old_r)


def linear_primality_test(n: int) -> int:
    """
    test for all odd value div between 3 and sqrt(n) if  div | n

    :param n: an integer to test for primality
    :return: True in and only if n is prime
    """
    if n <= 1:
        return False
    if n == 2 or n == 3:
        return True
    if not (n % 2):
        return False
    else:
        lim = int(math.sqrt(n)) + 1
        for i in range(3, lim + 1, 2):
            if n % i == 0:
                return False

        return True


def linear_prime_factorization(n: int) -> int:
    """
    used to speed up factorization when n is small

    :param n: an integer to be factorized
    :return: a prime factor of n
    """

    limit = int(math.sqrt(n)) + 2
    for i in range(2, limit):
        if n % i == 0:
            return i

    return -1


def gcd(a: int, b: int) -> int:
    """
    return the greatest common divisor of integers a,b
    """

    # here this is correct
    if a < 0:
        a = -a
    if b < 0:
        b = -b

    return extended_euclidean_division(a, b).gcd


def lcm(a: int, b: int) -> int:
    """
    return the lower common multiple of integers a,b
    """

    # TODO: is this correct here ?
    if a < 0:
        a = -a
    if b < 0:
        b = -b

    return extended_euclidean_division(a, b).lcm


def jacobi_symbol(u: int, v: int) -> int:
    """
    source: https://core.ac.uk/download/pdf/82664209.pdf

    computes the Jacobi symbol (u/v) of integers u,v
    :param u: an integer
    :param v: a strictly positive integer
    :return: the jacobi symbol
    """

    # TODO: implement the enhanced version stated in the paper

    if v <= 0:
        print(f"v must be positive, v: {v}")
        raise ValueError
    if v % 2 == 0:
        print(f"v must be odd, v: {v}")
        raise ValueError

    # some border cases
    if (u, v) == (0, 1):
        return 1
    if u == 0 and v > 1:
        return 0

    # some immediate cases
    if u == -1:
        return pow(-1, (v - 1) // 2)
    if u == 2:
        return pow(-1, (v * v - 1) // 8)

    # general case
    # two utilities for the general case
    def jac_minus1(x: int) -> int:
        return pow(-1, (x - 1) // 2)

    def jac2(x: int) -> int:
        return pow(-1, (x * x - 1) // 8)

    t = 1
    while True:
        # final case
        if u == 0:
            if v == 1:
                return t
            else:
                return 0

        if u < 0:
            u = -u
            t = t * jac_minus1(v)
            continue

        if u % 2 == 0:
            u = u // 2
            t = t * jac2(v)
            continue

        if u < v:
            u, v = v, u
            t = t * pow(-1, ((v - 1) // 2) * ((u - 1) // 2))
            continue

        # modulo case, there are two methods here
        # eisenstein method
        # b = u // v if (u // v) % 2 == 0 else (u // v) + 1
        # lebesgue method
        b = u // v

        # update u with b chosen by method
        u = u - b * v


def miller_rabin_witness(n: int, a: int) -> bool:
    """
    performs a test that indentifies a as a witness to n being composite

    source:  Introduction to algorithms
             Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein

    :param n: an integer to test for compositness
    :param a: an integer as the base for the test
    :return: True if and only if a is a witness for n compositness
    """

    # trim even values
    if not (n % 2):
        print(f"n must be odd: {n}")
        raise ValueError

    # find t, u such that n - 1 = 2**t * u and u is odd
    power2 = decompose_as_power_of_two(n - 1)
    t, u = power2.r, power2.d

    x = pow(a, u, n)

    for _ in range(t):
        y = pow(x, 2, n)

        if y == 1 and x != 1 and x != (n - 1):
            return True

        x = y

    if x != 1:
        return True
    return False


def miller_rabin_test(n: int, k: int) -> bool:
    """
    perform the miller rabin test k times on integer n

    source:  Introduction to algorithms
             Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein

    :param n: an integer to test for primality
    :param k: an integer number of times the test is to be performed
    :return: False if n is n composite, True if n is prime with possibility 1 - (1/2)**k
    """

    global PRIMES

    # base cases
    if n in PRIMES:
        return True

    if not (n % 2):
        return False

    a_values = {}
    for _ in range(k):

        # pick new value of a
        a = random.randint(1, n - 1)

        # be sure that the value has not been picked before
        while a in a_values:
            a = random.randint(1, n - 1)

        # store the new value to know that it must not be picked later
        a_values[a] = 0

        if miller_rabin_witness(n, a):
            return False

    return True


def solovay_strassen_test(n: int, k: int) -> bool:
    """
    perform the solovay strassen test on integer n k times
    k should be picked accordingly

    source: https://en.wikipedia.org/wiki/Solovay%E2%80%93Strassen_primality_test
    :param n: an integer to test primality
    :param k: number of times the test if performed
    :return: False if n is composite True if n is prime with confidence 1 - (1/2)**k
    """

    global PRIMES

    if n in PRIMES:
        return True

    if n == 2:
        return True
    if n % 2 == 0:
        return False

    if k < 1:
        return True
    else:
        # for big numbers this fails to be fast
        # a_values = random.sample([x for x in range(2, n)], k) if k < (n-2) else [x for x in range(2, n)]

        # this iterative form is better
        a_values = {}  # use dictionary for fast membership test
        for _ in range(k):

            # pick a not previously chosen integer for 'a'
            a = random.randint(1, n - 1)
            while a in a_values:
                a = random.randint(1, n - 1)
            a_values[a] = 0

            # jacobi symbol
            jac = jacobi_symbol(a, n)

            # power modulo n
            power = pow(a, (n - 1) // 2, n)

            # test equality
            equal_modulo = equality_modulo(jac, power, n)

            # if the test is failed return certainly False
            if jac == 0 or not equal_modulo:
                return False

        # if all values fail then return n "probably" prime
        PRIMES[n] = 0
        return True


def pollard_rho_naive(n: int, limit=None) -> int:
    i = 1
    x = random.randint(0, n - 1)
    y = x
    k = 2

    if limit is None:
        limit = n // 2

    while i <= limit:
        i += 1
        x = (pow(x, 2, n) - 1) % n

        d = gcd(y - x, n)

        if d != 1 and d != n:
            return d

        if i == k:
            y = x
            k *= 2

    return -1


def naive_factorize_rho_method(n: int, k_primality=10, primality_test=None, rho_limit=50, rho_tries=10):
    # go find the primes
    global PRIMES

    print(
        f"INFO: trying to factorize {n}, k_primality: {k_primality}, rho_limit: {rho_limit}, rho tries limit: {rho_tries}, primality_test: {primality_test}")

    if primality_test is None:
        print("INFO: primality test is None, set to miller rabin test")
        primality_test = partial(miller_rabin_test, k=k_primality)

    factors = {}

    # n is an small prime
    if n in PRIMES:
        factors[n] = 1
        return factors

    if not (n % 2):
        power2 = decompose_as_power_of_two(n)
        factors[2] = power2.r

        original_n = n
        n = power2.d
    else:
        original_n = n

    # n is a power of 2:
    if n == 1:
        return factors

    to_factorize = [n]
    stop_condition = len(to_factorize) == 0

    while not stop_condition:
        print("-" * 50)
        print(f"INFO: values to factorize: {to_factorize}\n      factors found: {factors}")

        value = to_factorize.pop(-1)

        # print(f"value is {value}")

        print(f"INFO: testing if {value} is prime, k limit set to {k_primality}")
        is_prime = primality_test(value)

        if not is_prime:

            print(f"INFO: {value} is not prime, trying to factorize with pollard rho method")

            factor = pollard_rho_naive(value)

            accum = 0
            while factor == -1 and accum < rho_tries:
                accum += 1

                print(f"    try {accum} to factorize {value} with pollard rho method")

                factor = pollard_rho_naive(value, rho_limit)

            print(f"INFO: factor of {value} found: {factor}")

            if factor == 1 or factor == -1:
                print("ERROR: pollard rho could not find non trivial factor")
                raise Exception
            else:
                p = factor
                q = value // p

                # test if p or q are primes
                if primality_test(p):
                    print(f"INFO: factor {p} found to be prime, updating factors dictionary")

                    if p in factors:
                        factors[p] += 1
                    else:
                        factors[p] = 1
                else:
                    print(f"INFO: factor {p} is not prime inserting into factorize list")
                    bisect.insort(to_factorize, p)

                if primality_test(q):
                    print(f"INFO: factor {q} found to be prime, updating factors dictionary")

                    if q in factors:
                        factors[q] += 1
                    else:
                        factors[q] = 1
                else:
                    print(f"INFO: factor {q} is not prime inserting into factorize list")
                    bisect.insort(to_factorize, q)

        else:

            print(f"INFO: {value} is prime, updating factors dictionary")

            if value in factors:
                factors[value] += 1
            else:
                factors[value] = 1

        stop_condition = len(to_factorize) == 0

    good_factorization = verify_factorization(original_n, factors)

    print("-"*120)
    print(f"INFO: factorization finished (factorization is good: {good_factorization}), returning: {factors}")
    print("-"*120)

    return factors


def find_factor_fermat_naive(n: int) -> Tuple[int, int]:
    """
    find a non trivial factor of n
    :param n: an integer to be factorized
    :return: a non trivial factor of n or -1 if fails
    """

    if not(n % 2):
        print(f"n must be odd ! n: {n}")
        raise ValueError

    global PRIMES

    if n in PRIMES:
        return -1, -1
    else:
        # test this to be sure of not looping a perfect square
        if is_perfect_square(n) is not None:
            sqrt_n = int(math.sqrt(n))
            return sqrt_n, sqrt_n

        # initialize both a and b
        a = int(math.sqrt(n)) + 1
        b = int(pow(a, 2)) - n

        # try to find a factor and update a,b values
        while a < n and is_perfect_square(b) is None:

            print(f"n: {n}, a: {a}, b: {b}")

            b += 2*a + 1
            a += 1

        if is_perfect_square(b) is not None:
            return a - int(math.sqrt(b)), a + int(math.sqrt(b))
        else:
            return -1, -1


def verify_factorization(n: int, factors: Dict[int, int]) -> bool:
    r = 1
    for factor in factors:
        r *= pow(factor, factors[factor])

    return n == r


def quadratic_sieve(n: int) -> int:
    """
    find a non trivial factor of n if n is not prime and not even
    :param n: an odd integer to factorize
    :return:  a non trivial factor of n
    """

    if not(n % 2):
        print(f"n must be odd: {n}")
        raise ValueError

    pass










