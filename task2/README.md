#Find the Ducks
În viața reală există mai multe tipuri de gândire. Unii oameni gândesc în imagini, unii în cuvinte, unii în concepte etc. De aceea "gândește înainte să spui ceva" nu funcționează pentru toată lumea... Dacă pui un coechipier să își imagineze o rață, cum va arăta rezultatul? (v. aphantasia test)

Noi am încercat să punem un LLM să-și imagineze rața Nitro în diverse contexte și am decis ca rezultatul va fi unul dintre task-urile voastre de azi.

Train data
Setul de date de antrenare este format din două părți

1. o serie de imagini tip .png
Imaginile sunt numerotate de la 1 la dimensiunea setului și fiecare dintre aceste ele poate conține maxim odată rața Nitro care poate fi rotită, răsturnată sau redimensionată.

2. un fișier de tip .csv care conține următoarele 4 coloane:
"DatapointID" = id-ul imaginii pe care o descrie (corespunde cu numele fișierului fără extensie)
"DuckOrNoDuck" = are valoare 1 dacă rața apare în imagine sau 0 în caz contrar
"PixelCount" = numărul de pixeli care formează rața (va fi 0 în cazul în care rața nu apare)
"BoundingBox" = 4 numere întregi separate prin spațiu care reprezintă în ordine coordonatele colțurilor stânga-sus și dreapta-jos ale bounding box-ului (toate 4 vor fi 0 în cazul în care rața nu apare)
Formal bounding box-ul este definit prin coordonatele x1 y1 x2 y2 unde:

x1 și y1 sunt maxim posibile
x2 și y2 sunt minim posibile
orice pixel de coordonate x, y care face parte din rață respectă x1 <= x <= x2 și y1 <= y <= y2
Test data
Setul de date de test va conține doar o serie de imagini tip .png numerotate de la 1 la dimensiunea setului.

Subtask-uri
Subtask 1 -- duck or no duck
Punctaj: 20p

Pentru primul subtask trebuie să identificați dacă rața Nitro apare sau nu în fiecare imagine. Veți fi punctați bazat pe acuratețe.

Subtask 2 -- pixel count
Punctaj: 35p

Trebuie calculați numărul de pixeli care fac parte din forma raței. Veți fi punctați în funcție de cât de departe sunteți de răspunsul real.

În cazul în care rața nu apare, răspunsul va fi 0

Subtask 3 -- bounding box
Punctaj: 45p

Trebuie să găsiți bounding box-ul raței, dacă aceasta există și să-l afișați în formatul descris la datele de antrenament. Dacă rața nu apare, trebuie afișate 4 valori de 0.

Veți fi punctați în funcție de raportul în care bounding box-ul calculat de voi se suprapune cu cel real.

Format output
Fișierul .csv pe care îl încărcați trebuie să conțină 4 coloane cu exact același format ca și datele de antrenament (vezi descrierea de mai sus și Sample Output). Toate coloanele trebuie să fie prezente și în ordinea cerută chiar dacă vrei să răspunzi pentru un singur subtask.
