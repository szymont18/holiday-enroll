# holiday-enroll
System for choosing the best time for holiday taking into account user preferences

# Jak używać
uruchom:
```commandline
python main.py
```
A następnie kieruj się promptem wypisywnym w konsoli

# Schemat pliku wejściowego
Przykładowe pliki wejściowe, można zobaczyć w folderze tests. 
<br> Wnętrze takiego pliku wygląda nasępująco:
```json
{
    "D1" : DZIEŃ_STARTOWY (int),
    "D2" : DZIEŃ_KOŃCZOWY (int),
    "alpha" : WAŻNOŚĆ CENY W FUNKCJI KOSZTU , IM WYŻSZA TYM CENA JEST WAŻNIEJSZA (float),
    "max_per_interval" : MAKSYMALNY_PIORYRTET_MOŻLIWY_DLA_DNIA (int),
    "max_seats" : MAKSYMALNA_LICZBA_OSÓB_KTÓRA_MOŻE_JECHAĆ (int),
    "min_days" : MINIMALNA_LICZBA_DNI_NA_KTÓRĄ_MOŻNA_JECHAĆ (int),
    "number_of_people": LICZBA_OSÓB (int),
    "prices" : CENY_POSZCZEGÓLNYCH_DNI_Z_PRZEDZIAŁU_D1-D2 (lista intów, o długości D2-D1),
    "F" : { 
        OSOBA - PRIORYTETTY_NA KAŻDY_DZIEŃ (int - lista intów o długości D2-D1, 
                                            gdzie wartości są z zakresu od 0 do 
                                            max_per_interval lub -1000)     
      }PRIORYTETY_DLA_KAŻDEJ_OSOBY
}
```
UWAGA NA ZAŁOŻENIA NIEZBĘDNE DO POPRAWNOŚCI:
<li> D2 > D1  //dzień końcowy dalszy niż początkowy
<li> min_days <= D2-D1 //minimalna liczba dni jest maksymalnie długości całego przedziału wyjazdowego
<li> len(prices) = D2-D1  //ceny są podane dokładnie dla tylu dni, ile jest w przedziale wyjazdowym
<li> len(F.keys()) = number_of_people  //w słowniku priorytetów uwzględniona jest każa osoba
<li> len(F[jakakolwiek osoba]) = D2-D1  //każda osoba ma uzupełnione piorytety na wszystkie dni z przedziału
<li> priorytety są z przedziału od 0 do max_per_interval, lub -1000 w wypadku niemożliwośći pojechania




[//]: # ()
[//]: # (User select dates between which would like to go to holiday. )

[//]: # (For example&#40;D1 = 0, D2 = 60&#41;.<br>)

[//]: # (Every User select priority for each day. For example:)

[//]: # ("Ala" select priority = [1,2,5,2,5,2, ..., 2,8,8,8,8,2,1]. )

[//]: # (|priority| = D2 -D1 = 60)