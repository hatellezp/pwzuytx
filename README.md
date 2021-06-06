# Description

__pwzuytx__ stands for [kryptos](https://en.wikipedia.org/wiki/Kryptos),
greek word meaning hidden.
Surprise this repository is about cryptography !

For the moment I'm reading anything I can gather about several systems
of cryptography.
Padding, cyphers, prime theory and such. Whenever I think I
understood something sufficiently then an implementation is made.

Must of the algorrithms are at the naive stade. Even more, python is 
not the best language to make stuff fast. Nevertheless, an
algorithm well made is good in any language.


## Modules

Each module is a file for the moment. If this project grows then I 
will make a more structured project.

* caesar:
  Implements [Caesar's cipher](https://en.wikipedia.org/wiki/Caesar_cipher)
  with some generalizations:
  * any alphabet;
  * preserve or not uppercase;
  * preserve or not unknown characters;
  * any shift's length;
* prime:
  Some utilities to work with prime numbers:
  * decomposition as a multiplication of a power of two and an odd
    integer;
  * [extended euclidean division](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm),
    nice tool to find several remarkable numbers: gcd, lcm...
  * a naive primality test (trial division up to **sqrt(n)**);
  * [jacobi_symbol](https://en.wikipedia.org/wiki/Jacobi_symbol) 
    important for some primality test, this [paper](https://core.ac.uk/download/pdf/82664209.pdf) is interesing
    if you want to read it;
  * [miller-rabin test](https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test)
    the miller rabin primality test, be aware, this is a 
    *probabilistic* primality test;
  * [solovay-strassen test](https://en.wikipedia.org/wiki/Solovay%E2%80%93Strassen_primality_test)
     another *probabilistic* primality test;
  * factorization with the [Pollard's rho](https://en.wikipedia.org/wiki/Pollard%27s_rho_algorithm)
   
  
## What's next:
* quadratic sieve 
* elliptic curves 
* automatize several types of attacks (collision...)

    