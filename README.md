Model
=====

## Embedding

Attribute | # of perceptron | description
----------|-----------------|------------
amount    | 1               | number of card with unique name
Name      | 1               | name of card. reduced to one perceptron as phrase.
Mana Cost | 1               |
Type line | # of types      |
power/toughness | 2 (optional) | If is a Creature.
loyality | 1 (optional) | If it is a Planeswalker.
Oracle text | # of phrases (optional) | Remove text in __r'\(.*\)'__ . 


to Pool Size.


## Inputlayer

Vector size from embedding, pool size.

## Hiddenlayer

Number of hidden layer is 45.


## Outputlayer

* 4 times of pool size
* future: Individual amount times of exists Card

UI filter criterias
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
