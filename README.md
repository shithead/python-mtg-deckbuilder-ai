Model
=====

## Inputlayer

Attribute | # of perceptron | description
----------|-----------------|------------
amount    | 1               | number of card with unique name
Name      | 1               | name of card. reduced to one perceptron as phrase.
Mana Cost | 1               |
Type line | # of types      |
power/toughness | 2 (optional) | If is a Creature.
loyality | 1 (optional) | If it is a Planeswalker.
Oracle text | # of phrases (optional) | Remove text in __r'\(.*\)'__ . 

## Hiddenlayer

Number of hidden layer is the half of numbers of attributes card attributes in Inputlayer.


## Outputlayer

* 60 perceptron 
* maximal 4 of same card
* card idententification is  $${{C_{idx}} \\over {UC + 1}} {\\pm  {1 \\over 2({UC + 1}) }}$$


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
