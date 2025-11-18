### 1. Implikace: Nashovo Equilibrium $\implies$ Maxmin = Minimax

Nechť $(\pi_1^*, \pi_2^*)$ je Nashovo equilibrium. Z definice NE plyne, že ani jeden hráč se nemůže jednostranně zlepšit.

**Pro hráče 1 (maximalizuje výhru):**
$$u_1(\pi_1^*, \pi_2^*) \ge u_1(\pi_1, \pi_2^*) \quad \forall \pi_1 \implies u_1(\pi_1^*, \pi_2^*) = \max_{\pi_1} u_1(\pi_1, \pi_2^*)$$

**Pro hráče 2 (maximalizuje svou výhru = minimalizuje výhru Hráče 1):**
$$u_2(\pi_1^*, \pi_2^*) \ge u_2(\pi_1^*, \pi_2) \iff -u_1(\pi_1^*, \pi_2^*) \ge -u_1(\pi_1^*, \pi_2)$$
$$u_1(\pi_1^*, \pi_2^*) \le u_1(\pi_1^*, \pi_2) \quad \forall \pi_2 \implies u_1(\pi_1^*, \pi_2^*) = \min_{\pi_2} u_1(\pi_1^*, \pi_2)$$

**Závěr:**
Spojením obou nerovností dostáváme:
$$u_1(\pi_1, \pi_2^*) \le u_1(\pi_1^*, \pi_2^*) \le u_1(\pi_1^*, \pi_2)$$
Hodnota hry $v = u_1(\pi_1^*, \pi_2^*)$ je tedy současně maximem minim (pro H1) a minimem maxim (pro H2).


### 2. Implikace: Maxmin = Minimax $\implies$ Nashovo Equilibrium

Nechť platí, že hra má hodnotu $v$, kde se maxmin rovná minimaxu, a strategie $\pi_1^*, \pi_2^*$ tuto hodnotu realizují:
$$\max_{\pi_1} \min_{\pi_2} u_1(\pi_1, \pi_2) = \min_{\pi_2} \max_{\pi_1} u_1(\pi_1, \pi_2) = v = u_1(\pi_1^*, \pi_2^*)$$

Musíme ukázat, že nikdo nechce uhnout:

1.  **Hráč 1:** Strategie $\pi_1^*$ mu garantuje alespoň $v$. Pokud by existovala lepší strategie $\pi_1'$, musela by zvýšit výhru proti *optimální* obraně soupeře, což je ve sporu s tím, že $v$ je `minimax` (horní závora pro Hráče 1). Tedy $u_1(\pi_1^*, \pi_2^*) \ge u_1(\pi_1, \pi_2^*)$.
2.  **Hráč 2:** Strategie $\pi_2^*$ garantuje, že Hráč 1 nezíská více než $v$. Kdyby Hráč 2 změnil strategii na $\pi_2'$, umožnil by Hráči 1 získat více (nebo stejně), což by pro Hráče 2 znamenalo nižší užitek. Tedy $u_2(\pi_1^*, \pi_2^*) \ge u_2(\pi_1^*, \pi_2')$.

**Závěr:**
Protože ani jeden hráč nemůže jednostrannou změnou strategie zvýšit svůj užitek, je dvojice $(\pi_1^*, \pi_2^*)$ Nashovým equilibriem.

---

### Shrnutí

Ve hrách s nulovým součtem platí:
$$(\pi_1^*, \pi_2^*) \text{ je NE} \iff u_1(\pi_1, \pi_2^*) \le u_1(\pi_1^*, \pi_2^*) \le u_1(\pi_1^*, \pi_2)$$