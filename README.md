# HPC KANNet - Rust implementation of the KANNet module
[![Srpski](img.shields.io)](README.md)
[![English](img.shields.io)](README.en.md)

Originalna struktura modula: https://github.com/Lija321/KANNet

#### Osnovne informacije
Student: Dejan Lisica
Projekat će biti implementiran za ocenu 10.
#### Opis problema
Projekat se fokusira na implementaciju 2D konvolucionog bloka koji se, za razliku od klasičnog pristupa zasnovanog na teoremi univerzalne aproksimacije, oslanja na Kolmogorov–Arnoldovu teoremu reprezentacije. Gotovo sve savremene mreže za prepoznavanje slika koriste konvolucione blokove; međutim, KANNet blok umesto linearnih kombinacija koristi niz ReLU funkcija kako bi za svaki blok aproksimirao višepromeljivu funkciju, omogućavajući učenje složenijih odnosa između piksela.

Naša arhitektura se zasniva na KAN sloju predloženom u radu:  
https://arxiv.org/abs/2406.02075  

Parametri $e$ i $s$ predstavljaju parametre koji se uče tokom treniranja. Srž svakog bloka je sloj mreže definisan sledećom dekompozicijom:

- $A = \text{ReLU}(E - x^T)$
- $B = \text{ReLU}(x^T - S)$
- $D = r \times A \cdot B$
- $F = D \cdot D$
- $y = \mathbf{W} \bigotimes F$

Naš blok može biti proizvoljne veličine $n$, za koju konstruišemo odgovarajući sloj mreže sa $n^2$ ulaza. Isto kao i kod klasičnog konvolucionog bloka, blok „klizi“ preko slike, a odgovarajući pikseli se mapiraju na ulaze sloja.

Motivacija ovog projekta nije evaluacija tačnosti klasifikacije ili performansi treniranja, već analiza računarskih karakteristika i performansi KAN-baziranih konvolucionih blokova, sa posebnim fokusom na aspekte visokoperformansnog računarstva (HPC).

#### Opis implementacije
Ulaz u KANNet blok biće nasumično generisane slike sa jednim kanalom. Kao i u slučaju klasične konvolucije, biće razmatrane različite veličine ulaza. Sve ulazne matrice i parametri biće nasumično generisani na početku eksperimenta i sačuvani u fajlove, čime se obezbeđuje da sekvencijalne i paralelne implementacije rade nad identičnim podacima.

Za svaku prostornu poziciju klizećeg prozora (dimenzija $n \times n$), odgovarajući pikseli se spljoštavaju u vektor $x \in \mathbb{R}^{n^2}$. Ovaj vektor predstavlja ulaz u KAN sloj, nakon čega se primenjuju prethodno definisane operacije kako bi se izračunala izlazna vrednost za tu prostornu lokaciju.

Strategije popunjavanja ivica (padding) biće razmotrene kako bi se obezbedilo korektno ponašanje na granicama slike i kontrolisala dimenzionalnost izlaza, analogno klasičnim konvolucionim blokovima. Izlaz KAN bloka može opciono biti praćen dodatnim operacijama kao što su:
- element-wise aktivacija (po potrebi),
- sažimanje (npr. maksimalno ili prosečno sažimanje),  
u cilju uporedivosti sa standardnim konvolucionim cevovodima.

Primarni fokus implementacije biće na efikasnosti, obrascima pristupa memoriji i strategijama paralelizacije, a ne na numeričkoj optimizaciji ili stabilnosti treniranja.

#### Sekvencijalno rešenje
Sekvencijalna implementacija će se striktno oslanjati na matematičku definiciju KAN sloja i mehanizam klizećeg prozora opisan iznad. Različite veličine ulaza i veličine bloka $n$ biće testirane, a vremena izvršavanja će biti merena i analizirana.

Referentna implementacija biće realizovana u jeziku Rust, sa posebnim akcentom na:
- eksplicitno upravljanje memorijom,
- keš-prijateljske rasporede podataka,
- izbegavanje nepotrebnih alokacija.

Ova sekvencijalna verzija predstavljaće osnovu za sva dalja poređenja performansi.

#### Paralelno rešenje
Paralelno rešenje biće zasnovano na prostornoj dekompoziciji ulazne matrice. Ulazna slika biće podeljena na nezavisne podregije, pri čemu će svaku regiju obrađivati zasebna nit. Posebna pažnja biće posvećena tome da svaki klizeći prozor u potpunosti pripada jednoj podregiji, kako bi se izbegle zavisnosti između niti.

Ključni aspekti strategije paralelizacije uključuju:
- podelu slike tako da nijedan KAN blok ne prelazi granice particija,
- pažljivo rukovanje padding regionima,
- minimizaciju troškova sinhronizacije.

Implementacija će koristiti niti i sinhronizacione primitive jezika Rust samo tamo gde je neophodno, uz težnju ka „embarrassingly parallel“ izvršavanju. Uticaj veličine bloka $n$, veličine ulaza i broja niti na performanse biće sistematski ispitan.

#### Eksperimenti jakog i slabog skaliranja
Biće sprovedeni eksperimenti jakog i slabog skaliranja:
- **Jako skaliranje:** veličina ulaza je fiksna, dok se broj niti povećava, pri čemu se meri ubrzanje i efikasnost.
- **Slabo skaliranje:** veličina ulaza se proporcionalno povećava sa brojem niti, kako bi se ispitalo da li se vreme izvršavanja zadržava približno konstantnim.

Rezultati će biti analizirani u skladu sa Amdalovim i Gustafsonovim zakonom, uz isticanje ograničenja i potencijala paralelizacije KAN-baziranih konvolucionih blokova.

#### Vizualizacija rezultata
Rezultati će biti vizualizovani pomoću grafika koji prikazuju:
- vreme izvršavanja u odnosu na veličinu ulaza za sekvencijalne i paralelne implementacije,
- ubrzanje i efikasnost u funkciji broja niti,
- poređenja različitih veličina bloka $n$.

Za vizualizaciju u Rust-u koristiće se jedna od sledećih biblioteka:
- `Plotters`
- `Rerun`

Vizualizacije će omogućiti jasnu analizu performansi i pružiti uvid u skalabilnost i računarsko ponašanje KANNet blokova u HPC kontekstu.
