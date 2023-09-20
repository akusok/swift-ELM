//
//  ELM.swift
//  test-ELM
//
//  Created by Anton Akusok on 29/07/2018.
//  Copyright © 2018 Anton Akusok. All rights reserved.
//

import Foundation
import MetalPerformanceShaders


class ELM {
    var c: Int? = nil
    let bsize: Int
    let bK: Int
    let bL: Int
    let alpha: Float
    let dataB: Float
    var dataBiasVector: MPSVector? = nil
    let dataStd: Float
    let W: [MPSMatrix]
    let bias: [MPSVector]
    var L: [[MPSMatrix?]]
    var B: [MPSMatrix]? = nil
    var batchX: MPSMatrix? = nil  // batch of X matrix on device
    var batchY: MPSMatrix? = nil
    var H: [MPSMatrix]? = nil
    let commandQueue: MTLCommandQueue

    
    init?(bsize: Int, bK: Int, bL: Int, alpha: Float, W: [URL], bias: [URL], dataB: Float?, dataBias: [Float]?, dataStd: Float?) {
        guard bK >= 1 && bL >= 1 else { print("bK and bL must be positive."); return nil }
        guard alpha >= 0 else { print("Alpha must be positive."); return nil }
        guard W.count == bK else { print("Wrong length of W."); return nil }
        guard bias.count == bK else { print("Wrong length of bias."); return nil }
        guard !(dataB != nil && dataBias != nil) else { print("Cannot set both dataB and dataBias."); return nil }
        
        self.bsize = bsize
        self.bK = bK
        self.bL = bL
        self.alpha = alpha
        self.W = W.map { loadNpyToMatrix(contentsOf: $0, mode: .storageModeManaged) }
        self.bias = bias.map { loadNpyToVector(contentsOf: $0, mode: .storageModeManaged) }
        self.dataB = dataB ?? 0.0
        self.dataStd = dataStd ?? 1.0
        if let dataBias = dataBias {
            let dataBiasBuffer = device.makeBuffer(bytes: dataBias, length: dataBias.count * MemoryLayout<Float>.stride, options: [.storageModeManaged])!
            self.dataBiasVector = MPSVector(buffer: dataBiasBuffer, descriptor: MPSVectorDescriptor(length: dataBias.count, dataType: .float32))
        }
        
        if let commandQueue = device.makeCommandQueue() {
            self.commandQueue = commandQueue
        } else {
            print("Cannot create command queue.")
            return nil
        }
        self.L = Array(repeating: Array(repeating: nil, count: bK), count: bK)
    }

    func _reset(inputs: Int, targets: Int) {
        // prepare ELM for training, initialize remaining matrices after we know data parameters
        c = targets
        H = Array(0 ..< bK).map { _ in MPSZeros(rows: bsize, columns: bL) }
        B = Array(0 ..< bK).map { _ in MPSZeros(rows: bL, columns: targets) }
        for row in 0 ..< bK {
            L[row][row] = MPSEye(bL, alpha: alpha)
            for col in 0 ..< row {
                L[row][col] = MPSZeros(bL)
            }
        }
        let rowBytesX = MPSMatrixDescriptor.rowBytes(fromColumns: inputs, dataType: .float32)
        let rowBytesY = MPSMatrixDescriptor.rowBytes(fromColumns: inputs, dataType: .float32)
        let bufferX = device.makeBuffer(length: bsize * rowBytesX, options: [.storageModePrivate])!
        let bufferY = device.makeBuffer(length: bsize * rowBytesY, options: [.storageModePrivate])!
        batchX = MPSMatrix(buffer: bufferX, descriptor: MPSMatrixDescriptor(rows: bsize, columns: inputs, rowBytes: rowBytesX, dataType: .float32))
        batchY = MPSMatrix(buffer: bufferY, descriptor: MPSMatrixDescriptor(rows: bsize, columns: targets, rowBytes: rowBytesY, dataType: .float32))
    }
    
    func fit(X: MPSMatrix, Y: MPSMatrix) {
        self._reset(inputs: X.columns, targets: Y.columns)
        for k in stride(from: 0, to: X.rows, by: bsize) {
            _batch_update(X: X, Y: Y, offset: k)
            print(k)
        }
        self.solve()
    }
    
    func partial_fit(X: MPSMatrix, Y: MPSMatrix) {
        if H == nil {
            self._reset(inputs: X.columns, targets: Y.columns)
        }

        print("Processing batch: ", separator: "", terminator: " ")
        for k in stride(from: 0, to: X.rows, by: bsize) {
            _batch_update(X: X, Y: Y, offset: k)
            print(k, separator: "", terminator: ", ")
        }
        print()
    }
    
    func _batch_update(X: MPSMatrix, Y: MPSMatrix, offset: Int) {
        // update current ELM with a portion of new data
        guard L[0][0] != nil, let c=c, let B=B, let H=H, let batchX=batchX, let batchY=batchY else { print("Error: un-initialized ELM encountered in training."); return }
        var cbuf: MTLCommandBuffer

        // prepare batch of data
        let batchRows = min(X.rows - offset, bsize)
        let copyOffset = MPSMatrixCopyOffsets(sourceRowOffset: UInt32(offset), sourceColumnOffset: 0, destinationRowOffset: 0, destinationColumnOffset: 0)
        let uploadX = MPSMatrixCopy(device: device, copyRows: batchRows, copyColumns: X.columns, sourcesAreTransposed: false, destinationsAreTransposed: false)
        let uploadY = MPSMatrixCopy(device: device, copyRows: batchRows, copyColumns: Y.columns, sourcesAreTransposed: false, destinationsAreTransposed: false)

        //  MEMORY LEAK WHILE LOADING DATA, STILL HERE!
//        cbuf = commandQueue.makeCommandBuffer()!
//        uploadX.encode(commandBuffer: cbuf, copyDescriptor: MPSMatrixCopyDescriptor(sourceMatrix: X, destinationMatrix: batchX, offsets: copyOffset))
//        uploadY.encode(commandBuffer: cbuf, copyDescriptor: MPSMatrixCopyDescriptor(sourceMatrix: Y, destinationMatrix: batchY, offsets: copyOffset))
//        cbuf.commit()
//        cbuf.waitUntilCompleted()
        
        // prepare kernels
        let matLoadBatch = MPSMatrixNeuron(device: device)
        matLoadBatch.setNeuronType(.linear, parameterA: 1.0/dataStd, parameterB: -dataB, parameterC: 0.0)
        
        let matMulXW = MPSMatrixMultiplication(device: device, resultRows: bsize, resultColumns: bL, interiorColumns: X.columns)
        let matMulHtH = MPSMatrixMultiplication(device: device, transposeLeft: true, transposeRight: false, resultRows: bL, resultColumns: bL, interiorColumns: bsize, alpha: 1.0, beta: 1.0)
        let matMulHtT = MPSMatrixMultiplication(device: device, transposeLeft: true, transposeRight: false, resultRows: bL, resultColumns: c, interiorColumns: bsize, alpha: 1.0, beta: 1.0)
        let matBiasTanh = MPSMatrixNeuron(device: device)
        matBiasTanh.setNeuronType(.tanH, parameterA: 1.0, parameterB: 1.0, parameterC: 0.0)
        
        //  normalize uploaded data
        cbuf = commandQueue.makeCommandBuffer()!
        matLoadBatch.encode(commandBuffer: cbuf, inputMatrix: batchX, biasVector: dataBiasVector, resultMatrix: batchX)
        cbuf.commit()

        // load batch data to GPU (normalizing on the way)
        if batchRows < bsize {  // reset batchX to zeros for incomplete batch of X
            cbuf = commandQueue.makeCommandBuffer()!
            let blit = cbuf.makeBlitCommandEncoder()!
            blit.fill(buffer: batchX.data, range: batchX.rowBytes*batchRows ..< batchX.matrixBytes, value: 0)
            blit.endEncoding()
            cbuf.commit()
        }

        // encode and run computations
        for i in 0 ..< bK {
            cbuf = commandQueue.makeCommandBuffer()!
            matMulXW.encode(commandBuffer: cbuf, leftMatrix: batchX, rightMatrix: W[i], resultMatrix: H[i])
            cbuf.commit()

            cbuf = commandQueue.makeCommandBuffer()!
            matBiasTanh.encode(commandBuffer: cbuf, inputMatrix: H[i], biasVector: bias[i], resultMatrix: H[i])
            cbuf.commit()

            cbuf = commandQueue.makeCommandBuffer()!
            matMulHtT.encode(commandBuffer: cbuf, leftMatrix: H[i], rightMatrix: batchY, resultMatrix: B[i])
            for j in 0 ... i {
                matMulHtH.encode(commandBuffer: cbuf, leftMatrix: H[i], rightMatrix: H[j], resultMatrix: L[i][j]!)
            }
            cbuf.commit()
        }
    }
    
    func solve() {
        // Batch Cholesky decomposition + solver
        guard L[0][0] != nil, let B=B, let c=c else { print("Error: un-initialized ELM encountered in solving."); return }
        
        // setup kernels
        let runCho = MPSMatrixDecompositionCholesky(device: device, lower: true, order: bL)
        let updateL1 = MPSMatrixSolveTriangular(device: device, right: true, upper: false, transpose: true, unit: false, order: bL, numberOfRightHandSides: bL, alpha: 1.0)
        let updateL2 = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true, resultRows: bL, resultColumns: bL, interiorColumns: bL, alpha: -1.0, beta: 1.0)
        var cbuf: MTLCommandBuffer!

        // batch Cholesky solver
        for i in 0 ..< bK {
            cbuf = commandQueue.makeCommandBuffer()!
            runCho.encode(commandBuffer: cbuf, sourceMatrix: L[i][i]!, resultMatrix: L[i][i]!, status: nil)
            cbuf.commit()
            
            cbuf = commandQueue.makeCommandBuffer()!
            for j in i+1 ..< bK {
                updateL1.encode(commandBuffer: cbuf, sourceMatrix: L[i][i]!, rightHandSideMatrix: L[j][i]!, solutionMatrix: L[j][i]!)
            }
            cbuf.commit()
            
            cbuf = commandQueue.makeCommandBuffer()!
            for j in i+1 ..< bK {
                for k in j ..< bK {
                    updateL2.encode(commandBuffer: cbuf, leftMatrix: L[k][i]!, rightMatrix: L[j][i]!, resultMatrix: L[k][j]!)
                }
            }
            cbuf.commit()
        }
        
        // setup kernels
        let solveTri1 = MPSMatrixSolveTriangular(device: device, right: false, upper: false, transpose: false, unit: false, order: bL, numberOfRightHandSides: c, alpha: 1.0)
        let solveTri2 = MPSMatrixSolveTriangular(device: device, right: false, upper: false, transpose: true, unit: false, order: bL, numberOfRightHandSides: c, alpha: 1.0)
        let updateT = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: bL, resultColumns: c, interiorColumns: bL, alpha: -1.0, beta: 1.0)
        let updateT2 = MPSMatrixMultiplication(device: device, transposeLeft: true, transposeRight: false, resultRows: bL, resultColumns: c, interiorColumns: bL, alpha: -1.0, beta: 1.0)
        
        // forward substitution
        for i in 0 ..< bK {
            cbuf = commandQueue.makeCommandBuffer()!
            solveTri1.encode(commandBuffer: cbuf, sourceMatrix: L[i][i]!, rightHandSideMatrix: B[i], solutionMatrix: B[i])
            cbuf.commit()
            
            cbuf = commandQueue.makeCommandBuffer()!
            for j in i+1 ..< bK {
                updateT.encode(commandBuffer: cbuf, leftMatrix: L[j][i]!, rightMatrix: B[i], resultMatrix: B[j])
            }
            cbuf.commit()
        }
        
        // backward substitution
        for i in (0 ..< bK).reversed() {
            cbuf = commandQueue.makeCommandBuffer()!
            solveTri2.encode(commandBuffer: cbuf, sourceMatrix: L[i][i]!, rightHandSideMatrix: B[i], solutionMatrix: B[i])
            cbuf.commit()
            
            cbuf = commandQueue.makeCommandBuffer()!
            for j in 0 ..< i {
                updateT2.encode(commandBuffer: cbuf, leftMatrix: L[i][j]!, rightMatrix: B[i], resultMatrix: B[j])
            }
            cbuf.commit()
        }
        
        cbuf.waitUntilCompleted()
        
        // clear memory
        batchX = nil
        batchY = nil
        H = nil
        L = Array(repeating: Array(repeating: nil, count: bK), count: bK)
    }

    func predict(X: MPSMatrix) -> MPSMatrix? {
        guard let c=c, let H=H else { print("Error: un-initialized ELM encountered in prediction."); return nil }
        guard let B=B else { print("Error: non-solved ELM encountered in prediction."); return nil }
        
        // prepare data storage
        let Yh = MPSZeros(rows: X.rows, columns: c)
        
        // prepare kernels
        let matMulXW = MPSMatrixMultiplication(device: device, resultRows: X.rows, resultColumns: bL, interiorColumns: X.columns)
        let matBiasTanh = MPSMatrixNeuron(device: device)
        matBiasTanh.setNeuronType(.tanH, parameterA: 1.0, parameterB: 1.0, parameterC: 0.0)
        let matMulHB = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: X.rows, resultColumns: c, interiorColumns: bL, alpha: 1.0, beta: 1.0)
        
        var cbuf: MTLCommandBuffer!
        
        // encode and run computations
        cbuf = commandQueue.makeCommandBuffer()!
        for i in 0 ..< bK {
            matMulXW.encode(commandBuffer: cbuf, leftMatrix: X, rightMatrix: W[i], resultMatrix: H[i])
        }
        cbuf.commit()
        
        cbuf = commandQueue.makeCommandBuffer()!
        for i in 0 ..< bK {
            matBiasTanh.encode(commandBuffer: cbuf, inputMatrix: H[i], biasVector: bias[i], resultMatrix: H[i])
        }
        cbuf.commit()
        
        for i in 0 ..< bK {
            cbuf = commandQueue.makeCommandBuffer()!
            matMulHB.encode(commandBuffer: cbuf, leftMatrix: H[i], rightMatrix: B[i], resultMatrix: Yh)
            cbuf.commit()
        }
        
        cbuf.waitUntilCompleted()
        return Yh
    }
    

}
