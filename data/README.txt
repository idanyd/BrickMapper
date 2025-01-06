
colors: available colors
elements: color variants of parts (inventory_parts)
inventories: inventories can contain Sets(sets.csv) and/or Collections(inventory_sets.csv) and/or Parts (inventory_parts.csv) and/or Minifigs (inventory_minifigs.csv)
inventory_minifigs: inventory ids of minifigs, to link to inventories.csv
inventory_parts: parts, color and quantity, by inventory id (each entry corresponds to a specific element (part + color) in a set. The parts breakdown in the manual is, in effect, a collection of inventory_parts)
inventory_sets: sets that are a collection of other sets
minifigs: minifigs
part_categories: available categories parts can belong to
part_relationships: rel_types are (P)rint, Pai(R), Su(B)-Part, (M)old, Pa(T)tern, (A)lternate.
parts: LEGO pieces, by id and name. No color information
sets: LEGO sets, by ID and name
themes: available themes
set_steps: steps associated with each set, by inventory_id
step_elements: elements in each step

* Sets and Minifigs contain one or more Inventories (inventories.csv)