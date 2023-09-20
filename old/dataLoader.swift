//
//  dataLoader.swift
//  desktop-ELM
//
//  Created by Anton Akusok on 03/08/2018.
//  Copyright © 2018 Anton Akusok. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

let MAGIC_PREFIX = "\u{93}NUMPY"

public enum NpyLoaderError: Error {
    case ParseFailed(message: String)
    case UnsupportedFormat(message: String)
}

public enum Endian: String {
    case host = "="
    case big = ">"
    case little = "<"
    case na = "|"
    
    static var all: [Endian] {
        return [.host, .big, .little, .na]
    }
}

public enum DataType: String {
    case uint8 = "u1"
    case uint16 = "u2"
    case uint32 = "u4"
    
    case int8 = "i1"
    case int16 = "i2"
    
    case float16 = "f2"
    case float32 = "f4"
    
    static var all: [DataType] {
        return [.uint8, .uint16, .uint32, .int8, .int16, .float16, .float32]
    }
    
    static let mpsType: [DataType: MPSDataType] = [.int8: .int8, .int16: .int16,
                                                   .uint8: .uInt8, .uint16:.uInt16, .uint32:.uInt32,
                                                   .float16:.float16, .float32:.float32]
    
    static let stride: [DataType: Int] = [.int8: 1, .int16: 2, .uint8: 1, .uint16: 2, .uint32: 4, .float16: 2, .float32: 4]
}


public func loadNpy(contentsOf url: URL, device: MTLDevice, verbose: Bool = true) throws -> MPSMatrix {
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    return try loadNpyFromData(data: data, device: device, verbose: verbose)
}


public func loadNpyFromData(data: Data, device: MTLDevice, verbose: Bool) throws -> MPSMatrix {
    guard let magic = String(data: data.subdata(in: 0..<6), encoding: .ascii) else {
        throw NpyLoaderError.ParseFailed(message: "Can't parse prefix")
    }
    guard magic == MAGIC_PREFIX else {
        throw NpyLoaderError.ParseFailed(message: "Invalid prefix: \(magic)")
    }
    
    let major = data[6]
    guard major == 1 || major == 2 else {
        throw NpyLoaderError.ParseFailed(message: "Invalid major version: \(major)")
    }
    
    let minor = data[7]
    guard minor == 0 else {
        throw NpyLoaderError.ParseFailed(message: "Invalid minor version: \(minor)")
    }
    
    let headerLen: Int
    let headerStarts: Int
    switch major {
    case 1:
        let tmp = data[8...9].withUnsafeBytes { UInt16(littleEndian: $0.pointee) }
        headerLen = Int(tmp)
        headerStarts = 10
    case 2:
        let tmp = data[8...11].withUnsafeBytes { UInt32(littleEndian: $0.pointee) }
        headerLen = Int(tmp)
        headerStarts = 12
    default:
        fatalError("Never happens.")
    }
    
    let headerData = data.subdata(in: headerStarts ..< headerStarts+headerLen)
    guard let str = String(data: headerData, encoding: .ascii) else {
        throw NpyLoaderError.ParseFailed(message: "Failed to load header")
    }

    let descr: String
    let endian: Endian
    let dataType: DataType
    let isFortranOrder: Bool
    do {
        let separate = str.components(separatedBy: CharacterSet(charactersIn: ", ")).filter { !$0.isEmpty }

        guard let descrIndex = separate.index(where: { $0.contains("descr") }) else {
            throw NpyLoaderError.ParseFailed(message: "Header does not contain the key 'descr'")
        }
        descr = separate[descrIndex + 1]

        guard let e = Endian.all.filter({ descr.contains($0.rawValue) }).first else {
            throw NpyLoaderError.ParseFailed(message: "Unknown endian")
        }
        endian = e

        guard let dt = DataType.all.filter({ descr.contains($0.rawValue) }).first else {
            throw NpyLoaderError.UnsupportedFormat(message: "Unsupported dtype: \(descr)")
        }
        dataType = dt

        guard let fortranIndex = separate.index(where: { $0.contains("fortran_order") }) else {
            throw NpyLoaderError.ParseFailed(message: "Header does not contain the key 'fortran_order'")
        }

        isFortranOrder = separate[fortranIndex+1].contains("True")
        guard !isFortranOrder else {
            throw NpyLoaderError.UnsupportedFormat(message: "Fortran-ordered data is unsupported.")
        }
    }

    var shape: [Int] = []
    do {
        guard let left = str.range(of: "("),
            let right = str.range(of: ")") else {
                throw NpyLoaderError.ParseFailed(message: "Shape not found in header.")
        }

        let substr = str[left.upperBound..<right.lowerBound]

        let strs = substr.replacingOccurrences(of: " ", with: "")
            .components(separatedBy: ",")
            .filter { !$0.isEmpty }
        for s in strs {
            guard let i = Int(s) else {
                throw NpyLoaderError.ParseFailed(message: "Shape contains invalid integer: \(s)")
            }
            shape.append(i)
        }
    }
    guard shape.count == 2 else {
        throw NpyLoaderError.UnsupportedFormat(message: "Only 2-dimensional matrices are supported.")
    }
    
    if verbose {
        print("Loading \(shape) matrix of \(dataType) elements, \(endian) endian and in \(isFortranOrder ? "Fortran" : "C")-ordering.")
    }
    
    let rows = shape[0]
    let columns = shape[1]
    let dtype = DataType.mpsType[dataType]!
    let rowBytes = columns * DataType.stride[dataType]!
    let content = data.suffix(from: headerStarts+headerLen)

    guard content.count == rows * rowBytes else {
        throw NpyLoaderError.ParseFailed(message: "Incorrect data size: get \(content.count) data bytes for \(rows) rows and \(columns) columns of type \(dataType).")
    }
    
    let matDescriptor = MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: rowBytes, dataType: dtype)
    let matBuffer = content.withUnsafeBytes { device.makeBuffer(bytes: $0, length: content.count, options: [])! }

    return MPSMatrix(buffer: matBuffer, descriptor: matDescriptor)
}
