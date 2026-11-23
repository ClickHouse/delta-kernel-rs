//! Estimates of the heap memory owned by expressions, so that an engine can bound the memory of a
//! cache that keeps them.

use std::collections::HashMap;
use std::io;
use std::mem::size_of;
use std::sync::Arc;

use serde::Serialize;

use super::{ColumnName, Expression, FieldTransform, Predicate, Scalar, Transform};
use crate::schema::DataType;

/// The number of bytes of heap memory that a value owns, not counting the value itself.
pub trait RetainedHeapSize {
    /// Returns the number of heap bytes owned by `self`, saturating at `usize::MAX`. Types that
    /// are not visited field by field are estimated by the length of their JSON serialization,
    /// and a value that cannot be serialized reports `usize::MAX`.
    fn retained_heap_size(&self) -> usize;
}

/// The size of the JSON serialization of a value, used for types whose internals are not visited.
/// It is at least the length of the names and literals the value owns.
fn serialized_size<T: Serialize + ?Sized>(value: &T) -> usize {
    struct CountingWriter(usize);

    impl io::Write for CountingWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0 += buf.len();
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let mut writer = CountingWriter(0);
    match serde_json::to_writer(&mut writer, value) {
        Ok(()) => writer.0,
        Err(_) => usize::MAX,
    }
}

fn vec_size<T: RetainedHeapSize>(values: &Vec<T>) -> usize {
    values
        .iter()
        .fold(values.capacity() * size_of::<T>(), |total, value| {
            total.saturating_add(value.retained_heap_size())
        })
}

impl RetainedHeapSize for String {
    fn retained_heap_size(&self) -> usize {
        self.capacity()
    }
}

impl<T: RetainedHeapSize> RetainedHeapSize for Box<T> {
    fn retained_heap_size(&self) -> usize {
        size_of::<T>().saturating_add(self.as_ref().retained_heap_size())
    }
}

impl<T: RetainedHeapSize> RetainedHeapSize for Arc<T> {
    fn retained_heap_size(&self) -> usize {
        (2 * size_of::<usize>() + size_of::<T>()).saturating_add(self.as_ref().retained_heap_size())
    }
}

impl<T: RetainedHeapSize> RetainedHeapSize for Option<T> {
    fn retained_heap_size(&self) -> usize {
        self.as_ref()
            .map_or(0, RetainedHeapSize::retained_heap_size)
    }
}

impl RetainedHeapSize for ColumnName {
    fn retained_heap_size(&self) -> usize {
        self.iter()
            .fold(self.len() * size_of::<String>(), |total, name| {
                total.saturating_add(name.retained_heap_size())
            })
    }
}

impl RetainedHeapSize for DataType {
    fn retained_heap_size(&self) -> usize {
        match self {
            DataType::Primitive(_) => 0,
            _ => serialized_size(self),
        }
    }
}

impl RetainedHeapSize for Scalar {
    fn retained_heap_size(&self) -> usize {
        match self {
            Scalar::String(value) => value.capacity(),
            Scalar::Binary(value) => value.capacity(),
            Scalar::Null(data_type) => data_type.retained_heap_size(),
            Scalar::Struct(_) | Scalar::Array(_) | Scalar::Map(_) => serialized_size(self),
            Scalar::Integer(_)
            | Scalar::Long(_)
            | Scalar::Short(_)
            | Scalar::Byte(_)
            | Scalar::Float(_)
            | Scalar::Double(_)
            | Scalar::Boolean(_)
            | Scalar::Timestamp(_)
            | Scalar::TimestampNtz(_)
            | Scalar::Date(_)
            | Scalar::Decimal(_) => 0,
        }
    }
}

impl RetainedHeapSize for FieldTransform {
    fn retained_heap_size(&self) -> usize {
        vec_size(&self.exprs)
    }
}

impl RetainedHeapSize for HashMap<String, FieldTransform> {
    fn retained_heap_size(&self) -> usize {
        // One control byte per bucket of the hash table.
        let buckets = self.capacity() * (size_of::<(String, FieldTransform)>() + 1);
        self.iter().fold(buckets, |total, (name, transform)| {
            total
                .saturating_add(name.retained_heap_size())
                .saturating_add(transform.retained_heap_size())
        })
    }
}

impl RetainedHeapSize for Transform {
    fn retained_heap_size(&self) -> usize {
        self.input_path
            .retained_heap_size()
            .saturating_add(self.field_transforms.retained_heap_size())
            .saturating_add(vec_size(&self.prepended_fields))
    }
}

impl RetainedHeapSize for Expression {
    fn retained_heap_size(&self) -> usize {
        match self {
            Expression::Literal(value) => value.retained_heap_size(),
            Expression::Column(name) => name.retained_heap_size(),
            Expression::Predicate(predicate) => predicate.retained_heap_size(),
            Expression::Struct(fields, nullability) => {
                vec_size(fields).saturating_add(nullability.retained_heap_size())
            }
            Expression::Transform(transform) => transform.retained_heap_size(),
            Expression::Unary(unary) => unary.expr.retained_heap_size(),
            Expression::Binary(binary) => binary
                .left
                .retained_heap_size()
                .saturating_add(binary.right.retained_heap_size()),
            Expression::Variadic(variadic) => vec_size(&variadic.exprs),
            Expression::Opaque(opaque) => vec_size(&opaque.exprs),
            Expression::Unknown(name) => name.retained_heap_size(),
            Expression::ParseJson(parse_json) => parse_json
                .json_expr
                .retained_heap_size()
                .saturating_add(serialized_size(parse_json.output_schema.as_ref())),
            Expression::MapToStruct(map_to_struct) => map_to_struct.map_expr.retained_heap_size(),
        }
    }
}

impl RetainedHeapSize for Predicate {
    fn retained_heap_size(&self) -> usize {
        match self {
            Predicate::BooleanExpression(expr) => expr.retained_heap_size(),
            Predicate::Not(predicate) => predicate.retained_heap_size(),
            Predicate::Unary(unary) => unary.expr.retained_heap_size(),
            Predicate::Binary(binary) => binary
                .left
                .retained_heap_size()
                .saturating_add(binary.right.retained_heap_size()),
            Predicate::Junction(junction) => vec_size(&junction.preds),
            Predicate::Opaque(opaque) => vec_size(&opaque.exprs),
            Predicate::Unknown(name) => name.retained_heap_size(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::column_name;

    #[test]
    fn literal_strings_are_counted() {
        let short = Expression::literal("a");
        let long = Expression::literal("a".repeat(1000));
        assert!(long.retained_heap_size() >= 1000);
        assert!(long.retained_heap_size() > short.retained_heap_size());
    }

    #[test]
    fn transform_counts_its_expressions() {
        let empty = Expression::Transform(Transform::new_top_level());
        let with_value = Expression::Transform(
            Transform::new_top_level()
                .with_inserted_field(Some("a"), Arc::new(Expression::literal("x".repeat(500)))),
        );
        assert!(with_value.retained_heap_size() >= empty.retained_heap_size() + 500);
    }

    #[test]
    fn columns_are_counted() {
        let expr = Expression::column(column_name!("some_column_name"));
        assert!(expr.retained_heap_size() >= "some_column_name".len());
    }
}
