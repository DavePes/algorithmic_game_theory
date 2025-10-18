
### Chceme ukázat



Cílem je dokázat rovnost mezi výplatou v Nashově equlibriu a maximinovou hodnotou pro hráče 1.



$$u_1(\pi_1^*, \pi_2^*) = v_1 = \max_{\pi_1} \min_{\pi_2} u_1(\pi_1, \pi_2)$$


---
### Odvození z Nashova equilibra 

**Odvození z definice rovnováhy pro Hráče 2**



Vycházíme z podmínky pro Hráče 2:

$$u_2(\pi_1^*, \pi_2^*) \ge u_2(\pi_1^*, \pi_2')$$



Protože se jedná o hru s nulovým součtem, platí $u_2 = -u_1$. Můžeme tedy nerovnost přepsat:

$$-u_1(\pi_1^*, \pi_2^*) \ge -u_1(\pi_1^*, \pi_2')$$Vynásobením nerovnosti číslem -1 se otočí znaménko:$$u_1(\pi_1^*, \pi_2^*) \le u_1(\pi_1^*, \pi_2') \quad \forall \pi_2'$$


**Dostáváme tedy, že:**

$$u_1(\pi_1^*, \pi_2^*) = \min_{\pi_2} u_1(\pi_1^*, \pi_2)$$

---


**Odvození z definice rovnováhy pro Hráče 1**

Víme, že z podmínky pro Hráče 2 platí:

$$u_1(\pi_1^*, \pi_2^*) = \min_{\pi_2} u_1(\pi_1^*, \pi_2)$$

Z definice NE pro Hráče 1 víme:

$$u_1(\pi_1^*, \pi_2^*) \ge u_1(\pi_1', \pi_2^*) \quad \forall \pi_1'$$

**Dostáváme tedy, že:**
$$\max_{\pi_1} u_1(\pi_1, \pi_2^*) = u_1(\pi_1^*, \pi_2^*)$$

Nyní se podívejme na definici maximinové hodnoty

$$v_1 = \max_{\pi_1} \min_{\pi_2} u_1(\pi_1, \pi_2)$$


Dosazením tohoto do předchozí nerovnosti získáme:

$$v_1 = u_1(\pi_1^*, \pi_2^*)$$

Tím jsme ukázali, že naše strategie je správná.

### Odvození z maximin

Vyjdeme z definice maximin:

$$v_1 = \max_{\pi_1} \min_{\pi_2} u_1(\pi_1, \pi_2)$$

A chceme dokázat, že dané strategie splňují požadavky NE. Dokážeme pro $\pi_1$, pro $\pi_2$ by se dokazovalo obdobně.

Bereme maximální strategii přes všechny možné strategie, tedy platí, že:

$$u_1(\pi_1, \pi_2) \ge u_1(\pi_1', \pi_2) \quad \forall \pi_1'$$


