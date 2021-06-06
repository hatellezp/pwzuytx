import math
from typing import Tuple, Dict, List, Optional


class BPrimes:
    def __init__(self, b: Optional[int] = None, ps: Optional[List[int]] = None):

        if ps is not None:
            self.primes = ps
            self.b = len(ps)

        else:
            # go get the first b primes
            b_primes = []
            with open("/home/horacio/Langs/python/src/PRIMES_LIST.txt", 'r') as f:
                accum = 0
                while accum < b:
                    prime_str = f.readline()
                    b_primes.append(int(prime_str.strip()))
                    accum += 1

            self.b = b
            self.primes = b_primes

    def factorize(self, n: int) -> Tuple[int, List[int]]:

        # create dictionnary of factors
        factors = [0] * self.b

        # factorize if possible
        for index, prime_number in enumerate(self.primes):
            power = 0
            while not (n % prime_number):
                power += 1
                n = n // prime_number

            factors[index] = power

        return n, factors

    def aggregate(self, powers: List[int]) -> int:
        if len(powers) != self.b:
            return - 1
        else:
            accum = 1
            for i in range(len(powers)):
                accum *= pow(self.primes[i], powers[i])

            return accum

    def __str__(self):
        return f"BPrimes(b: {self.b}, primes: {self.primes})"

    @staticmethod
    def reduce_mod2(powers: List[int]) -> List[int]:
        return [x % 2 for x in powers]

    @staticmethod
    def sum_powers(powers1: List[int], powers2: List[int]) -> List[int]:
        if len(powers1) != len(powers2):
            print(f"powers list must have same length: len1: {len(powers1)}, len2: {len(powers2)}")
            raise ValueError

        return [powers1[i] + powers2[i] for i in range(len(powers2))]


def qs_data_gathering_phase(n: int, b: int, m: int):
    # discard even numbers
    if not(n % 2):
        print(f"n must be odd: {n}")
        raise ValueError

    # discard perfect squares
    if not(n % int(math.sqrt(n))):
        print(f"perfect square! n: {n}, sqrt: {int(math.sqrt(n))}")
        return int(math.sqrt(n))

    # initialize values
    # begin the data gathering phase
    x = int(math.sqrt(n)) + 1  # x
    bprimes = BPrimes(b=b)  # go search the first b primes
