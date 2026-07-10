# Knots to Knobs: Post-hoc Diversity Steering (Rozšíření a Evaluace)

Tento repozitář obsahuje rozšíření pro projekt **"Pulling the Right Levers: Steerable Collaborative Filtering with Sparse Autoencoders"** (originální repozitář naleznete na [anonymous.4open.science/r/knots-to-knobs](https://anonymous.4open.science/r/knots-to-knobs/README.md)).

Toto rozšíření implementuje automatickou grid sweep evaluaci steeringu diverzity a interaktivní HTML dashboard pro vizualizaci výsledků.

Link na report [https://smartgourd.github.io/recsys-sae-diversity-steering/].

---

## 📂 Obsah rozšíření (Nové soubory)

Pro zprovoznění tohoto rozšíření stačí zkopírovat (přidat) následující tři soubory do kořenového adresáře naklonovaného originálního projektu:

1. **`run_evaluation.py`** – Python skript pro kompletní evaluaci. Automaticky najde diverzifikační neurony v SAE na základě korelace s žánrovými profily a provede grid sweep (4 počty steered neuronů $N \in [1, 2, 3, 5]$ × 17 sil steeringu $S \in [0, 1, \dots, 100]$).
2. **`index.html`** – Interaktivní webová prezentace napsaná v čistém HTML/JavaScriptu za použití moderního skleněného designu (glassmorphism) a Pareto grafů. Obsahuje živé demo pro testovací profily uživatelů a přehledné srovnání.
3. **`results/steering_evaluation.json`** – Vygenerovaný datový soubor obsahující kompletní metriky z proběhlé evaluace (Recall@20, nDCG@20, Precision@20, Tag Diversity, Collaborative Filtering Diversity a Overlap s baseline).

---

## ⚙️ Jak to zprovoznit

### 1. Klonování a instalace originálního projektu
Nejprve naklonujte originální repozitář a připravte virtuální prostředí podle původního návodu:

```bash
git clone <url-originalniho-repozitare>
cd knots-to-knobs

# Vytvoření virtuálního prostředí a instalace závislostí
python -m venv .venv
source .venv/bin/activate  # na Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Přidání evaluačních souborů
Zkopírujte soubory `run_evaluation.py` a `index.html` (včetně složky `results/` s vygenerovaným JSONem, pokud ji máte) do kořenového adresáře projektu.

### 3. Spuštění evaluace
Pokud chcete přeměřit všechny konfigurace a vygenerovat čerstvá data, spusťte:

```bash
python run_evaluation.py
```
Skript využívá **workaround pro standardizační bug** v originálním souboru `sae.py`. Původní kód modifikuje vstupní tenzor in-place (`x -= x_mean`), což by bez ošetření zkreslovalo výpočet čisté ELSA baseline v rámci jednoho evaluačního cyklu. Náš skript před předáním do SAE tenzor klonuje (`z_input = batch.clone()`), což umožňuje čisté spuštění bez nutnosti modifikovat původní kód repozitáře.

*Poznámka ke složitosti:* Pro výpočet Tag Diversity se počítá průměrná párová kosinová vzdálenost mezi tagovými vektory doporučených filmů ($O(k^2)$ na uživatele). Pro zkrácení doby běhu ze serverových hodin na cca 2 minuty skript provádí evaluaci na reprezentativním vzorku **2000 náhodně vybraných testovacích uživatelů** (což dává statisticky stabilní průměry s odchylkou pod 0.001 vůči plné testovací sadě o 16 667 uživatelích).

### 4. Zobrazení interaktivního dashboardu
Pro prohlížení výsledků a interaktivní zkoušení doporučení spusťte lokální webový server v kořenovém adresáři:

```bash
# Pomocí Pythonu:
python -m http.server 3000

# Nebo pomocí Node.js / npx:
npx serve
```
Poté otevřete prohlížeč na adrese [http://localhost:3000/](http://localhost:3000/).

---

## 📈 Klíčová zjištění evaluace

* **Steerovatelnost vs. Přesnost**: Post-hoc steering pomocí SAE umožňuje plynule zvyšovat diverzitu doporučení bez nutnosti model přetrénovávat. Nejlepší poměr zisku diverzity k obětovanému Recallu (sweet spot s poměrem **1.2x až 1.9x**) nabízejí konfigurace s **1 až 2 steered neurony** při nízkých silách steeringu (S = 2 až 5).
* **Limitace extrémního steeringu**: Při agresivním steeringu (N = 5, S = 30+) dochází k potlačení personalizovaných rysů doporučení. Diverzita tagů sice vzroste o **+29.0%**, ale Recall klesá o dramatických **-45.9%** a překryv doporučení (Overlap) padá na **30.0%** – doporučení se stávají generickými.
* **Kritické srovnání s Čistou ELSA**:
  * **Čistá ELSA (bez SAE)** dosahuje výrazně lepších výsledků jak v přesnosti (Recall@20 = 0.4305, tj. o **+10.4%** více než SAE baseline), tak v přirozené diverzitě (Tag Diversity = 0.2877, tj. o **+2.0%** více než SAE baseline).
  * Samotné zařazení rekonstrukční vrstvy SAE způsobuje obrovskou degradaci. Pro překonání přirozené diverzity čisté ELSA (např. na 0.2908 u N=2, S=7, zisk +1.1% oproti čisté ELSA) musíme obětovat masivní část přesnosti – Recall@20 padá na 0.3767 (ztráta **-12.5%** oproti čisté ELSA).
  * **Závěr**: Pro praktické nasazení post-hoc steeringu v produkci je klíčové nejprve výrazně zlepšit věrnost rekonstrukce samotného SAE (např. doladěním hyperparametrů nebo trénováním na širším latentním prostoru).
