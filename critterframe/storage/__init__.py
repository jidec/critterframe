"""
Storage mechanics with no knowledge of any specific entity: records/ modules
supply the schema, these supply the reading and writing.

Three backends, each used for what it is good at, and which one a thing lands
in is a property of the data rather than a preference:

  imagestore -- LMDB, for original analysis images. One blob per occurrence,
                stored byte-exactly, read by key. A key-value store because
                that is the whole access pattern, and byte-exact because
                re-encoding an image is irreversible.
  tables     -- parquet, for occurrences and masks. Wide, columnar, read whole
                or by column, and rewritten as a unit -- a snapshot or a
                keyed merge, never an append.
  sqlite     -- for runs and metric values. Appended row by row over long
                interruptible runs, queried by key, and needing to survive a
                process dying halfway through, which is exactly what a
                rewrite-as-a-unit format cannot offer.
"""
