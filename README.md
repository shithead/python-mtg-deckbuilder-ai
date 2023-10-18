Criterias
================

* presets
    * aggro
    * control

* mana color
* avg. mana costs

Environments
==========

Deck
----
* equal effects and types
    * synagy
    * combos

* 1/3 +- 4/deck.size land cards
    * optimized to non basic lands


* Deck attributes
    * maximal amount of cards (60)
    * maximal amount of same card (4)

Fillup to 60 cards. calculate mana curve and avg. mana costs for numbers of basic lands.
Than optimize basic lands.

Sideboard
--------
* size: 15
    + oversize: 4/deck.size

Card
-----
* types/subtypes
    * instant
    * sorcery
    * enchantment
    * creature
    * planeswalker

* manacosts
* manacolor
    * different attributes

* amount

Pool
----

* all my available cards
* load json

Constructor
----------

* action
    * preview
    * next
    * drop
    * pick
