# Moves

All these methods are used to generate new proposals from the tree space. The
order of argument is always the node first, and the tree second. It is always
assumed that the `node` will point to a node contained within the `tree`, so
that *both* will be modified. The node that is targeted by the modification is
returned. For the `swap!` method, the *parent* node is given as argument.

## Moves to explore the trees space

```@docs
grow!
prune!
change!
swap!
```

## Tree modification utilities

```@docs
BART.createrule!
BART.collapse!
BART.update!
```